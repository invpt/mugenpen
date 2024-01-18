# mugenpen
Infinite canvas notetaking software for devices running Wayland with graphics styli a.k.a. pen tablets.

## Current status
Rendering is done with OpenGL. It only works on Wayland, because that's the environment I use and it was necessary to use low-level APIs due to the lack of support for pen tablet input in existing higher-level libraries for window creation. A major goal is to keep the dependencies light: it currently only has 46, which is quite low for a Rust application, and as a result it compiles very quickly.
