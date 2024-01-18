use glow::HasContext;
use std::{mem::size_of, os::raw::c_void};
use wayland_client::{
    delegate_noop, event_created_child,
    protocol::{
        wl_buffer, wl_callback, wl_compositor, wl_keyboard, wl_registry, wl_seat, wl_shm,
        wl_shm_pool, wl_surface, wl_touch,
    },
    Connection, Dispatch, Proxy, QueueHandle, WEnum,
};
use wayland_egl::WlEglSurface;

use wayland_protocols::{
    wp::tablet::zv2::client::{
        zwp_tablet_manager_v2, zwp_tablet_seat_v2, zwp_tablet_tool_v2, zwp_tablet_v2,
    },
    xdg::shell::client::{xdg_surface, xdg_toplevel, xdg_wm_base},
};

use khronos_egl as egl;

mod math;

use math::Vec2;

// required workaround for khronos-egl
#[link(name = "EGL")]
#[link(name = "GLESv2")]
extern "C" {}

struct GlobalsFinished;
struct Draw;

fn main() {
    let conn = Connection::connect_to_env().unwrap();

    let mut initial_eq = conn.new_event_queue();
    let initial_qh = initial_eq.handle();
    let mut eq = conn.new_event_queue::<State>();
    let qh = eq.handle();

    let display = conn.display();
    display.get_registry(&initial_qh, ());
    display.sync(&initial_qh, GlobalsFinished);

    let mut incomplete = IncompleteState {
        qh,
        running: true,
        completed: false,
        base_surface: None,
        wl_egl_surface: None,
        wm_base: None,
        xdg_surface: None,
        xdg_toplevel: None,
        configured: false,
        seat: None,
        manager: None,
        egl: None,
        gl: None,
        program: None,
        dim: (0, 0),
    };

    println!("Starting the example window app, press <ESC> to quit.");

    while !incomplete.completed {
        initial_eq.blocking_dispatch(&mut incomplete).unwrap();
    }

    let mut state = State {
        wl: WlState {
            running: incomplete.running,
            configured: incomplete.configured,
            base_surface: incomplete.base_surface.unwrap(),
            egl_surface: incomplete.wl_egl_surface.unwrap(),
            wm_base: incomplete.wm_base.unwrap(),
            xdg_surface: incomplete.xdg_surface.unwrap(),
            xdg_toplevel: incomplete.xdg_toplevel.unwrap(),
            seat: incomplete.seat.unwrap(),
            tablet_manager: incomplete.manager.unwrap(),
            dim: incomplete.dim,
        },
        strokes: vec![],
        egl: incomplete.egl.unwrap(),
        gl: incomplete.gl.unwrap(),
        program: incomplete.program.unwrap(),
        pan: (0.0, 0.0),
        tool_down: false,
        tool_pressure: 0,
        touch_down: None,
        touch_prev_loc: None,
    };

    state.wl.base_surface.frame(&incomplete.qh, Draw);
    state
        .egl
        .instance
        .swap_buffers(state.egl.display, state.egl.surface)
        .unwrap();
    state.wl.base_surface.commit();

    while state.wl.running {
        eq.blocking_dispatch(&mut state).unwrap();
    }
}

struct IncompleteState {
    qh: QueueHandle<State>,
    running: bool,
    configured: bool,
    completed: bool,
    base_surface: Option<wl_surface::WlSurface>,
    wl_egl_surface: Option<wayland_egl::WlEglSurface>,
    wm_base: Option<xdg_wm_base::XdgWmBase>,
    xdg_surface: Option<xdg_surface::XdgSurface>,
    xdg_toplevel: Option<xdg_toplevel::XdgToplevel>,
    seat: Option<wl_seat::WlSeat>,
    manager: Option<zwp_tablet_manager_v2::ZwpTabletManagerV2>,
    egl: Option<EglState>,
    gl: Option<glow::Context>,
    program: Option<glow::Program>,
    dim: (u32, u32),
}

struct EglState {
    instance: egl::Instance<egl::Static>,
    display: egl::Display,
    surface: egl::Surface,
}

struct State {
    wl: WlState,
    egl: EglState,

    gl: glow::Context,
    program: glow::Program,
    strokes: Vec<Stroke>,
    pan: (f32, f32),
    tool_down: bool,
    tool_pressure: u32,
    touch_down: Option<i32>,
    touch_prev_loc: Option<(f64, f64)>,
}

struct Stroke {
    vertex_count: usize,
    vertex_storage: Vec<StrokeVertex>,
    vao: glow::VertexArray,
    vbo: glow::NativeBuffer,
    prev_point: Option<Vec2>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct StrokeVertex {
    position: Vec2,
    normal: Vec2,
}

unsafe impl bytemuck::Zeroable for StrokeVertex {}
unsafe impl bytemuck::Pod for StrokeVertex {}

impl Stroke {
    fn new(gl: &glow::Context) -> Stroke {
        let vertex_storage = vec![<StrokeVertex as bytemuck::Zeroable>::zeroed(); 16];
        unsafe {
            let vao = gl.create_vertex_array().unwrap();
            gl.bind_vertex_array(Some(vao));
            let vbo = gl.create_buffer().unwrap();
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            gl.buffer_data_u8_slice(
                glow::ARRAY_BUFFER,
                bytemuck::cast_slice(&vertex_storage),
                glow::DYNAMIC_DRAW,
            );
            gl.vertex_attrib_pointer_f32(
                0,
                2,
                glow::FLOAT,
                false,
                size_of::<StrokeVertex>() as i32,
                0,
            );
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(
                1,
                2,
                glow::FLOAT,
                false,
                size_of::<StrokeVertex>() as i32,
                size_of::<Vec2>() as i32,
            );
            gl.enable_vertex_attrib_array(1);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            gl.bind_vertex_array(None);

            Stroke {
                vertex_count: 0,
                vertex_storage,
                vao,
                vbo,
                prev_point: None,
            }
        }
    }

    fn draw(&self, gl: &glow::Context) {
        unsafe {
            gl.bind_vertex_array(Some(self.vao));
            gl.draw_arrays(glow::TRIANGLE_STRIP, 0, self.vertex_count as i32);
            gl.bind_vertex_array(None);
        }
    }

    fn add_point(&mut self, gl: &glow::Context, point: Vec2, width: f32) {
        if let Some(prev_point) = self.prev_point {
            let pt_dir = width * (point - prev_point).normalize().perp();
            let left_pt = point + pt_dir;
            let right_pt = point - pt_dir;

            self.add_vertices(
                gl,
                [
                    StrokeVertex {
                        position: left_pt,
                        normal: pt_dir,
                    },
                    StrokeVertex {
                        position: right_pt,
                        normal: pt_dir,
                    },
                ]
                .into_iter(),
            );
        }

        self.prev_point = Some(point);
    }

    fn add_vertices(
        &mut self,
        gl: &glow::Context,
        mut vertices: impl ExactSizeIterator<Item = StrokeVertex>,
    ) {
        let new_count = self.vertex_count + vertices.len();
        if self.vertex_storage.len() < new_count {
            let mut new_len = self.vertex_storage.len();
            while new_len < new_count {
                new_len *= 2;
            }

            self.vertex_storage
                .resize(new_len, <StrokeVertex as bytemuck::Zeroable>::zeroed());

            self.vertex_storage[self.vertex_count..new_count]
                .fill_with(|| vertices.next().unwrap());

            unsafe {
                gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vbo));
                gl.buffer_data_u8_slice(
                    glow::ARRAY_BUFFER,
                    bytemuck::cast_slice(&*self.vertex_storage),
                    glow::DYNAMIC_DRAW,
                );
                gl.bind_buffer(glow::ARRAY_BUFFER, None);
            }
        } else {
            self.vertex_storage[self.vertex_count..new_count]
                .fill_with(|| vertices.next().unwrap());
            
            unsafe {
                gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vbo));
                gl.buffer_sub_data_u8_slice(
                    glow::ARRAY_BUFFER,
                    (self.vertex_count * size_of::<StrokeVertex>()) as i32,
                    bytemuck::cast_slice(&self.vertex_storage[self.vertex_count..new_count]),
                );
                gl.bind_buffer(glow::ARRAY_BUFFER, None);
            }
        }
        self.vertex_count = new_count;
    }
}

struct WlState {
    running: bool,
    configured: bool,
    base_surface: wl_surface::WlSurface,
    #[allow(dead_code)]
    egl_surface: wayland_egl::WlEglSurface,
    #[allow(dead_code)]
    wm_base: xdg_wm_base::XdgWmBase,
    #[allow(dead_code)]
    xdg_surface: xdg_surface::XdgSurface,
    #[allow(dead_code)]
    xdg_toplevel: xdg_toplevel::XdgToplevel,
    #[allow(dead_code)]
    seat: wl_seat::WlSeat,
    #[allow(dead_code)]
    tablet_manager: zwp_tablet_manager_v2::ZwpTabletManagerV2,
    dim: (u32, u32),
}

impl Dispatch<wl_registry::WlRegistry, ()> for IncompleteState {
    fn event(
        state: &mut Self,
        registry: &wl_registry::WlRegistry,
        event: wl_registry::Event,
        _: &(),
        conn: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let wl_registry::Event::Global {
            name, interface, ..
        } = event
        {
            match &*interface {
                "wl_compositor" => {
                    let compositor =
                        registry.bind::<wl_compositor::WlCompositor, _, _>(name, 1, &state.qh, ());
                    let surface = compositor.create_surface(qh, ());
                    state.base_surface = Some(surface);

                    state.init_egl(conn);

                    if state.wm_base.is_some() && state.xdg_surface.is_none() {
                        state.init_xdg_surface(&state.qh.clone());
                    }
                }
                "wl_seat" => {
                    state.seat =
                        Some(registry.bind::<wl_seat::WlSeat, _, _>(name, 1, &state.qh, ()));
                }
                "xdg_wm_base" => {
                    let wm_base =
                        registry.bind::<xdg_wm_base::XdgWmBase, _, _>(name, 1, &state.qh, ());
                    state.wm_base = Some(wm_base);

                    if state.base_surface.is_some() && state.xdg_surface.is_none() {
                        state.init_xdg_surface(&state.qh.clone());
                    }
                }
                "zwp_tablet_manager_v2" => {
                    state.manager = Some(
                        registry.bind::<zwp_tablet_manager_v2::ZwpTabletManagerV2, _, _>(
                            name,
                            1,
                            &state.qh,
                            (),
                        ),
                    );
                }
                _ => {}
            }
        }
    }
}

impl Dispatch<wl_callback::WlCallback, GlobalsFinished> for IncompleteState {
    fn event(
        state: &mut Self,
        _: &wl_callback::WlCallback,
        _: <wl_callback::WlCallback as wayland_client::Proxy>::Event,
        _: &GlobalsFinished,
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        state.manager.as_ref().unwrap().get_tablet_seat(
            state.seat.as_ref().unwrap(),
            &state.qh,
            (),
        );
        state.completed = true;
    }
}

// Ignore events from these object types in this example.
delegate_noop!(IncompleteState: ignore wl_compositor::WlCompositor);
delegate_noop!(IncompleteState: ignore wl_surface::WlSurface);
delegate_noop!(IncompleteState: ignore wl_shm::WlShm);
delegate_noop!(IncompleteState: ignore wl_shm_pool::WlShmPool);
delegate_noop!(IncompleteState: ignore wl_buffer::WlBuffer);

delegate_noop!(State: ignore wl_compositor::WlCompositor);
delegate_noop!(State: ignore wl_surface::WlSurface);
delegate_noop!(State: ignore wl_shm::WlShm);
delegate_noop!(State: ignore wl_shm_pool::WlShmPool);
delegate_noop!(State: ignore wl_buffer::WlBuffer);
delegate_noop!(State: zwp_tablet_manager_v2::ZwpTabletManagerV2);

impl IncompleteState {
    fn init_xdg_surface(&mut self, qh: &QueueHandle<State>) {
        let wm_base = self.wm_base.as_ref().unwrap();
        let base_surface = self.base_surface.as_ref().unwrap();

        let xdg_surface = wm_base.get_xdg_surface(base_surface, qh, ());
        let toplevel = xdg_surface.get_toplevel(qh, ());
        toplevel.set_title("mugenpen".into());

        self.xdg_surface = Some(xdg_surface);
        self.xdg_toplevel = Some(toplevel);
    }

    fn init_egl(&mut self, conn: &Connection) {
        let display_ptr = conn.display().id().as_ptr();
        let wl_egl_surface =
            WlEglSurface::new(self.base_surface.as_ref().unwrap().id(), 1280, 720).unwrap();
        let surface_ptr = wl_egl_surface.ptr();

        self.dim.0 = 1280;
        self.dim.1 = 720;

        let egl = egl::Instance::new(egl::Static);
        let egl_display = unsafe { egl.get_display(display_ptr as *mut c_void) }.unwrap();
        egl.initialize(egl_display).unwrap();
        let egl_config = egl
            .choose_first_config(
                egl_display,
                &[
                    egl::RED_SIZE,
                    8,
                    egl::GREEN_SIZE,
                    8,
                    egl::BLUE_SIZE,
                    8,
                    egl::ALPHA_SIZE,
                    8,
                    egl::NONE,
                ],
            )
            .unwrap()
            .unwrap();

        let egl_context = egl
            .create_context(
                egl_display,
                egl_config,
                None,
                &[
                    egl::CONTEXT_MAJOR_VERSION,
                    3,
                    egl::CONTEXT_MINOR_VERSION,
                    2,
                    egl::NONE,
                ],
            )
            .unwrap();

        let egl_surface = unsafe {
            egl.create_window_surface(egl_display, egl_config, surface_ptr as *mut _, None)
        }
        .unwrap();

        egl.make_current(
            egl_display,
            Some(egl_surface),
            Some(egl_surface),
            Some(egl_context),
        )
        .unwrap();

        self.init_gl(&egl);

        self.egl = Some(EglState {
            instance: egl,
            display: egl_display,
            surface: egl_surface,
        });
        self.wl_egl_surface = Some(wl_egl_surface);
    }

    fn init_gl(&mut self, egl: &egl::Instance<egl::Static>) {
        unsafe {
            let gl = glow::Context::from_loader_function(|s| {
                egl.get_proc_address(s).unwrap() as *const _
            });

            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);
            gl.enable(glow::BLEND);

            let vertex_shader_src = r#"
                #version 320 es
                precision mediump float;

                layout (location = 0) in vec2 aPos;
                layout (location = 1) in vec2 in_norm;

                out vec2 norm;
                out vec2 f_pos;
                out vec2 pos;

                uniform vec2 viewport;
                uniform vec2 pan;

                vec2 to_ndc(vec2 p);
                
                void main() {
                    gl_Position = vec4(to_ndc(vec2(aPos.x + pan.x, aPos.y + pan.y)), 0.0, 1.0);
                    norm = in_norm;
                    pos = aPos;
                    if (gl_VertexID % 2 == 0) {
                        f_pos = aPos - in_norm;
                    } else {
                        f_pos = aPos + in_norm;
                    }
                }

                vec2 to_ndc(vec2 p) {
                    return vec2(p.x / viewport.x * 2.0 - 1.0, -p.y / viewport.y * 2.0 + 1.0);
                }
            "#;

            let frag_shader_src = r#"
                #version 320 es
                precision mediump float;

                in vec2 norm;
                in vec2 f_pos;
                in vec2 pos;

                out mediump vec4 FragColor;

                uniform vec2 viewport;
                
                void main() {
                    float distance = abs(dot(pos - f_pos, norm)) / length(norm);
                    FragColor = vec4(1.0, 1.0, 1.0, length(norm) - distance);
                }
            "#;

            let shader_sources = [
                (glow::VERTEX_SHADER, vertex_shader_src),
                (glow::FRAGMENT_SHADER, frag_shader_src),
            ];

            let mut shaders = Vec::with_capacity(shader_sources.len());

            let program = gl.create_program().expect("Cannot create program");
            for (shader_type, shader_source) in shader_sources.iter() {
                let shader = gl
                    .create_shader(*shader_type)
                    .expect("Cannot create shader");
                gl.shader_source(shader, shader_source);
                gl.compile_shader(shader);
                if !gl.get_shader_compile_status(shader) {
                    panic!("{}", gl.get_shader_info_log(shader));
                }
                gl.attach_shader(program, shader);
                shaders.push(shader);
            }

            gl.link_program(program);
            if !gl.get_program_link_status(program) {
                panic!("{}", gl.get_program_info_log(program));
            }

            for shader in shaders {
                gl.detach_shader(program, shader);
                gl.delete_shader(shader);
            }

            self.gl = Some(gl);
            self.program = Some(program);
        }
    }
}

impl Dispatch<xdg_wm_base::XdgWmBase, ()> for State {
    fn event(
        _: &mut Self,
        wm_base: &xdg_wm_base::XdgWmBase,
        event: xdg_wm_base::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let xdg_wm_base::Event::Ping { serial } = event {
            wm_base.pong(serial);
        }
    }
}

impl Dispatch<xdg_surface::XdgSurface, ()> for State {
    fn event(
        state: &mut Self,
        xdg_surface: &xdg_surface::XdgSurface,
        event: xdg_surface::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let xdg_surface::Event::Configure { serial, .. } = event {
            xdg_surface.ack_configure(serial);
            state.wl.configured = true;
        }
    }
}

impl Dispatch<xdg_toplevel::XdgToplevel, ()> for State {
    fn event(
        state: &mut Self,
        _: &xdg_toplevel::XdgToplevel,
        event: xdg_toplevel::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let xdg_toplevel::Event::Close {} = event {
            state.wl.running = false;
        }
    }
}

impl Dispatch<wl_seat::WlSeat, ()> for State {
    fn event(
        _: &mut Self,
        seat: &wl_seat::WlSeat,
        event: wl_seat::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let wl_seat::Event::Capabilities {
            capabilities: WEnum::Value(capabilities),
        } = event
        {
            if capabilities.contains(wl_seat::Capability::Keyboard) {
                seat.get_keyboard(qh, ());
            }

            if capabilities.contains(wl_seat::Capability::Touch) {
                seat.get_touch(qh, ());
            }
        }
    }
}

impl Dispatch<wl_keyboard::WlKeyboard, ()> for State {
    fn event(
        state: &mut Self,
        _: &wl_keyboard::WlKeyboard,
        event: wl_keyboard::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let wl_keyboard::Event::Key { key, .. } = event {
            if key == 1 {
                state.wl.running = false;
            }
        }
    }
}

impl Dispatch<wl_touch::WlTouch, ()> for State {
    fn event(
        state: &mut Self,
        _: &wl_touch::WlTouch,
        event: <wl_touch::WlTouch as wayland_client::Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            wl_touch::Event::Down { id, x, y, .. } => {
                if state.touch_down.is_none() {
                    state.touch_down = Some(id);
                    state.touch_prev_loc = Some((x, y));
                }
            }
            wl_touch::Event::Up { id, .. } => {
                if state.touch_down == Some(id) {
                    state.touch_down = None;
                    state.touch_prev_loc = None;
                }
            }
            wl_touch::Event::Motion { id, x, y, .. } => {
                if state.touch_down == Some(id) {
                    if let Some((prev_x, prev_y)) = state.touch_prev_loc {
                        let dx = x - prev_x;
                        let dy = y - prev_y;

                        state.pan.0 += dx as f32;
                        state.pan.1 += dy as f32;
                    }

                    state.touch_prev_loc = Some((x, y));
                }
            }
            _ => {}
        }
    }
}

impl Dispatch<zwp_tablet_seat_v2::ZwpTabletSeatV2, ()> for State {
    fn event(
        _: &mut Self,
        _: &zwp_tablet_seat_v2::ZwpTabletSeatV2,
        _: <zwp_tablet_seat_v2::ZwpTabletSeatV2 as wayland_client::Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }

    event_created_child!(State, zwp_tablet_seat_v2::ZwpTabletSeatV2, [
        zwp_tablet_seat_v2::EVT_TABLET_ADDED_OPCODE => (zwp_tablet_v2::ZwpTabletV2, ()),
        zwp_tablet_seat_v2::EVT_TOOL_ADDED_OPCODE => (zwp_tablet_tool_v2::ZwpTabletToolV2, ()),
    ]);
}

impl Dispatch<zwp_tablet_v2::ZwpTabletV2, ()> for State {
    fn event(
        _: &mut Self,
        _: &zwp_tablet_v2::ZwpTabletV2,
        _: <zwp_tablet_v2::ZwpTabletV2 as wayland_client::Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<zwp_tablet_tool_v2::ZwpTabletToolV2, ()> for State {
    fn event(
        state: &mut Self,
        _: &zwp_tablet_tool_v2::ZwpTabletToolV2,
        event: <zwp_tablet_tool_v2::ZwpTabletToolV2 as wayland_client::Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            zwp_tablet_tool_v2::Event::Motion { x, y } if state.tool_down => {
                state.strokes.last_mut().unwrap().add_point(
                    &state.gl,
                    Vec2 {
                        x: x as f32 - state.pan.0,
                        y: y as f32 - state.pan.1,
                    },
                    (state.tool_pressure as f32 / u16::MAX as f32 * 5.0).max(1.0),
                );
            }
            zwp_tablet_tool_v2::Event::Pressure { pressure } => {
                state.tool_pressure = pressure;
            }
            zwp_tablet_tool_v2::Event::Down { .. } => {
                if let Some(last) = state.strokes.last() {
                    if last.vertex_count != 0 {
                        state.strokes.push(Stroke::new(&state.gl));
                    }
                } else {
                    state.strokes.push(Stroke::new(&state.gl));
                }
                state.tool_down = true;
            }
            zwp_tablet_tool_v2::Event::Up { .. } => {
                state.tool_down = false;
            }
            _ => (),
        }
    }
}

impl Dispatch<wl_callback::WlCallback, Draw> for State {
    fn event(
        state: &mut Self,
        _: &wl_callback::WlCallback,
        _: <wl_callback::WlCallback as Proxy>::Event,
        _: &Draw,
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        unsafe {
            state.gl.use_program(Some(state.program));

            // setup uniforms
            let pan_location = state.gl.get_uniform_location(state.program, "pan").unwrap();
            state
                .gl
                .uniform_2_f32(Some(&pan_location), state.pan.0, state.pan.1);
            let viewport_location = state
                .gl
                .get_uniform_location(state.program, "viewport")
                .unwrap();
            state.gl.uniform_2_f32(
                Some(&viewport_location),
                state.wl.dim.0 as f32,
                state.wl.dim.1 as f32,
            );

            // clear the screen
            state.gl.clear_color(0.0, 0.0, 0.0, 1.0);
            state.gl.clear(glow::COLOR_BUFFER_BIT);

            // draw the strokes!
            for stroke in &state.strokes {
                stroke.draw(&state.gl);
            }

            state.gl.use_program(None);
        }

        state.wl.base_surface.frame(qh, Draw);
        state
            .egl
            .instance
            .swap_buffers(state.egl.display, state.egl.surface)
            .unwrap();
        state.wl.base_surface.commit();
    }
}
