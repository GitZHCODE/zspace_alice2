#version 120

// Vertex attributes
attribute vec3 a_position;
attribute vec3 a_normal;
attribute vec3 a_color;

// Uniforms
uniform mat4 u_modelViewProjectionMatrix;
uniform mat4 u_modelViewMatrix;
uniform mat4 u_normalMatrix;

// Varying outputs to fragment shader
varying vec3 v_position;
varying vec3 v_normal;
varying vec3 v_color;

void main() {
    // Transform position to view space
    v_position = (u_modelViewMatrix * vec4(a_position, 1.0)).xyz;
    
    // Transform normal to view space
    v_normal = normalize((u_normalMatrix * vec4(a_normal, 0.0)).xyz);
    
    // Pass through vertex color
    v_color = a_color;
    
    // Transform position to clip space
    gl_Position = u_modelViewProjectionMatrix * vec4(a_position, 1.0);
}
