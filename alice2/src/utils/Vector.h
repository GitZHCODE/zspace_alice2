#pragma once

#ifndef ALICE2_VECTOR_H
#define ALICE2_VECTOR_H

#include <cmath>
#include <algorithm>

namespace alice2 {

    // 2D Vector class
    struct Vec2 {
        float x, y;

        Vec2() : x(0), y(0) {}
        Vec2(float x_, float y_) : x(x_), y(y_) {}

        // Basic operations
        Vec2 operator+(const Vec2& other) const { return Vec2(x + other.x, y + other.y); }
        Vec2 operator-(const Vec2& other) const { return Vec2(x - other.x, y - other.y); }
        Vec2 operator-() const { return Vec2(-x, -y); }  // Unary minus
        Vec2 operator*(float scalar) const { return Vec2(x * scalar, y * scalar); }
        Vec2 operator/(float scalar) const { return Vec2(x / scalar, y / scalar); }
        
    };

    // 3D Vector class
    struct Vec3 {
        float x, y, z;

        Vec3() : x(0), y(0), z(0) {}
        Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

        // Basic operations
        Vec3 operator+(const Vec3& other) const { return Vec3(x + other.x, y + other.y, z + other.z); }
        Vec3 operator-(const Vec3& other) const { return Vec3(x - other.x, y - other.y, z - other.z); }
        Vec3 operator-() const { return Vec3(-x, -y, -z); }  // Unary minus
        Vec3 operator*(float scalar) const { return Vec3(x * scalar, y * scalar, z * scalar); }
        Vec3 operator/(float scalar) const { return Vec3(x / scalar, y / scalar, z / scalar); }
        
        Vec3& operator+=(const Vec3& other) { x += other.x; y += other.y; z += other.z; return *this; }
        Vec3& operator-=(const Vec3& other) { x -= other.x; y -= other.y; z -= other.z; return *this; }
        Vec3& operator*=(float scalar) { x *= scalar; y *= scalar; z *= scalar; return *this; }

        bool operator==(const Vec3 &o) const
        {
            constexpr float EPS = 1e-6f;
            return fabs(x - o.x) < EPS && fabs(y - o.y) < EPS && fabs(z - o.z) < EPS;
        }
        bool operator!=(const Vec3 &other) const { return !(*this == other); }

        // Vector operations
        float dot(const Vec3& other) const { return x * other.x + y * other.y + z * other.z; }
        Vec3 cross(const Vec3& other) const { 
            return Vec3(y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x); 
        }
        
        float length() const { return std::sqrt(x * x + y * y + z * z); }
        float lengthSquared() const { return x * x + y * y + z * z; }
        
        Vec3 normalized() const { 
            float len = length(); 
            return len > 0 ? *this / len : Vec3(); 
        }
        
        void normalize() { 
            float len = length(); 
            if (len > 0) { x /= len; y /= len; z /= len; } 
        }

        float distanceTo(const Vec3& other) const{
            float dist = (*this - other).length();
            return dist; 
        }

        float angleBetween(const Vec3 &other) const
        {
            float len1 = length();
            float len2 = other.length();
            if (len1 == 0.0f || len2 == 0.0f)
            {
                // undefined; choose 0 to avoid division-by-zero
                return 0.0f;
            }
            float cosTheta = dot(other) / (len1 * len2);
            // clamp for numeric safety
            cosTheta = std::clamp(cosTheta, -1.0f, 1.0f);
            return std::acos(cosTheta);
        }

        // Access
        float& operator[](int index) { return (&x)[index]; }
        const float& operator[](int index) const { return (&x)[index]; }

        // Static utility functions
        static Vec3 lerp(const Vec3& a, const Vec3& b, float t) {
            return a + (b - a) * t;
        }
    };

    // Z-up coordinate system constants
    namespace ZUp {
        static const Vec3 FORWARD = Vec3(0, 1, 0);   // +Y forward
        static const Vec3 RIGHT = Vec3(1, 0, 0);     // +X right
        static const Vec3 UP = Vec3(0, 0, 1);        // +Z up
        static const Vec3 BACK = Vec3(0, -1, 0);     // -Y back
        static const Vec3 LEFT = Vec3(-1, 0, 0);     // -X left
        static const Vec3 DOWN = Vec3(0, 0, -1);     // -Z down
    }

} // namespace alice2

#endif // ALICE2_VECTOR_H
