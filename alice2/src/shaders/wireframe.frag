#version 120

// Varying inputs from vertex shader
varying vec3 v_color;

void main() {
    // Output vertex color directly
    gl_FragColor = vec4(v_color, 1.0);
}
