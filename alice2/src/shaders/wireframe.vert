#version 120

// Vertex attributes
attribute vec3 a_position;
attribute vec3 a_color;

// Uniforms
uniform mat4 u_modelViewProjectionMatrix;

// Varying outputs to fragment shader
varying vec3 v_color;

void main() {
    // Pass through vertex color
    v_color = a_color;
    
    // Transform position to clip space
    gl_Position = u_modelViewProjectionMatrix * vec4(a_position, 1.0);
}
