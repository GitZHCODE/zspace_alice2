#version 120

// Varying inputs from vertex shader
varying vec3 v_position;
varying vec3 v_normal;
varying vec3 v_color;

// Lighting uniforms
uniform vec3 u_lightDirection;
uniform vec3 u_lightColor;
uniform vec3 u_ambientLight;
uniform bool u_enableLighting;

void main() {
    vec3 finalColor = v_color;
    
    if (u_enableLighting) {
        // Normalize the interpolated normal
        vec3 normal = normalize(v_normal);
        
        // Calculate light direction (negate to point towards light)
        vec3 lightDir = normalize(-u_lightDirection);
        
        // Calculate diffuse lighting
        float diffuseFactor = max(dot(normal, lightDir), 0.0);
        
        // Combine ambient and diffuse lighting
        vec3 lighting = u_ambientLight + u_lightColor * diffuseFactor;
        
        // Apply lighting to vertex color
        finalColor = v_color * lighting;
    }
    
    // Output final color
    gl_FragColor = vec4(finalColor, 1.0);
}
