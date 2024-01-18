# mugenpen
Infinite canvas notetaking software for devices running Wayland with graphics styli a.k.a. pen tablets. It's intended to be a lightweight alternative to Xournal++ unhindered by a page-oriented flow.

## Current status
Rendering is done with OpenGL. Pressure-sensitive drawing and panning around the canvas is possible. You can't save or open files, and the window lacks decorations or any way to resize. It only works on Wayland, because that's the environment I use and it was necessary to use low-level APIs due to the lack of support for pen tablet input in existing higher-level libraries for window creation. A major goal is to keep the dependencies light: it currently only has 45, which is quite low for a Rust application, and as a result it compiles very quickly.
