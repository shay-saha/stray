#pragma once

#include "simulation.hpp"

#include <array>
#include <cmath>
#include <functional>

namespace wildlife::local {

inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline float length(Vec3 a) { return std::sqrt(dot(a, a)); }
inline Vec3 unit(Vec3 a) {
    const auto distance = length(a);
    return distance > 1e-7f ? a * (1.0f / distance) : Vec3{0.0f, 1.0f, 0.0f};
}
inline Vec3 rgb(std::uint32_t value) {
    return {static_cast<float>((value >> 16) & 255) / 255.0f,
            static_cast<float>((value >> 8) & 255) / 255.0f, static_cast<float>(value & 255) / 255.0f};
}

struct Vertex {
    Vec3 position;
    Vec3 normal;
    Vec3 color;
    float alpha = 1.0f;
    std::array<float, 4> markings{};
};

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;

    void clear() { vertices.clear(); indices.clear(); }
};

struct AnimalPose {
    Vec3 origin;
    float yaw;
    float phase;
    float moving;
    float feeding;
    float airborne;
    bool swimming;
    float scale;
    unsigned detail = 2;
};

using GroundHeight = std::function<float(float east, float north)>;

void build_animal(Mesh& mesh, const Species& species, AnimalProfile profile, AnimalPose pose, const GroundHeight& height);

} // namespace wildlife::local
