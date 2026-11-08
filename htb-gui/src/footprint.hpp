#pragma once

#include "globe.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>

namespace wildlife {

inline constexpr std::size_t range_samples = 24;

namespace footprint_detail {

inline constexpr float earth_radius_km = 6371.0f;
inline constexpr float maximum_extent = 1.35f;

inline const std::array<Vec2, range_samples>& bearings() {
    static const auto points = [] {
        std::array<Vec2, range_samples> result;
        for (std::size_t i = 0; i < range_samples; ++i) {
            const float angle = -2.0f * std::numbers::pi_v<float> * static_cast<float>(i) / range_samples;
            result[i] = {std::cos(angle), std::sin(angle)};
        }
        return result;
    }();
    return points;
}

struct RangeShape {
    std::size_t broad_phase;
    std::size_t fine_phase;
    float broad_amplitude;
    float fine_amplitude;

    explicit RangeShape(std::size_t colony_index) {
        std::uint32_t seed = static_cast<std::uint32_t>(colony_index) + 0x9e3779b9u;
        seed = (seed ^ (seed >> 16)) * 0x85ebca6bu;
        seed = (seed ^ (seed >> 13)) * 0xc2b2ae35u;
        seed ^= seed >> 16;
        broad_phase = (seed & 0xffu) % range_samples;
        fine_phase = ((seed >> 8) & 0xffu) % range_samples;
        broad_amplitude = 0.075f + 0.035f * static_cast<float>((seed >> 16) & 0xffu) / 255.0f;
        fine_amplitude = 0.025f + 0.020f * static_cast<float>(seed >> 24) / 255.0f;
    }

    Vec2 offset(std::size_t sample, float radius) const {
        const auto& directions = bearings();
        const float broad = directions[(2 * sample + broad_phase) % range_samples].x;
        const float fine = directions[(3 * sample + fine_phase) % range_samples].y;
        const float extent = radius * (1.17f + broad_amplitude * broad + fine_amplitude * fine);
        const Vec2 bearing = directions[sample];
        return {bearing.x * extent, bearing.y * extent};
    }
};

inline Vec3 unit_vector(Vec3 point) {
    const float inverse = 1.0f / std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
    return {point.x * inverse, point.y * inverse, point.z * inverse};
}

inline ImU32 with_alpha(ImU32 color, unsigned int alpha) {
    return (color & ~IM_COL32_A_MASK) | (alpha << IM_COL32_A_SHIFT);
}

// Foreshortened ranges can fold along the limb at subpixel scales. Recover the
// visible outline from the clipped samples in clockwise order for ImGui's fill.
inline std::size_t limb_outline(std::array<ImVec2, range_samples * 2>& points, std::size_t count) {
    std::sort(points.data(), points.data() + count, [](ImVec2 a, ImVec2 b) {
        return a.x < b.x || (a.x == b.x && a.y < b.y);
    });
    std::array<ImVec2, range_samples * 2 + 1> hull;
    std::size_t size = 0;
    const auto append = [&](ImVec2 point, std::size_t start) {
        while (size >= start + 2) {
            const ImVec2 a = hull[size - 2];
            const ImVec2 b = hull[size - 1];
            const float turn = (b.x - a.x) * (point.y - a.y) - (b.y - a.y) * (point.x - a.x);
            if (turn > 1.0e-6f) {
                break;
            }
            --size;
        }
        hull[size++] = point;
    };
    for (std::size_t i = 0; i < count; ++i) {
        append(points[i], 0);
    }
    const std::size_t upper_start = size - 1;
    for (std::size_t i = count - 1; i > 0; --i) {
        append(points[i - 1], upper_start);
    }
    --size; // The final point is the start of the closed outline.
    std::copy_n(hull.begin(), size, points.begin());
    return size;
}

} // namespace footprint_detail

inline std::array<Vec2, range_samples> colony_range_boundary(std::size_t index, float radius_km) {
    const footprint_detail::RangeShape shape(index);
    std::array<Vec2, range_samples> boundary;
    for (std::size_t i = 0; i < boundary.size(); ++i) {
        boundary[i] = shape.offset(i, radius_km);
    }
    return boundary;
}

struct ProjectedFootprint {
    std::array<ImVec2, range_samples * 2> points;
    std::size_t count = 0;
    ImVec2 center{};
    ImVec2 min{};
    ImVec2 max{};
    float radius_px = 0.0f;
    float facing = 0.0f;

    bool hit(ImVec2 pointer) const {
        const float tolerance = radius_px < 11.0f ? 2.5f : 1.0f;
        if (count < 3 || pointer.x < min.x - tolerance || pointer.x > max.x + tolerance ||
            pointer.y < min.y - tolerance || pointer.y > max.y + tolerance) {
            return false;
        }
        bool inside = false;
        for (std::size_t i = 0, previous = count - 1; i < count; previous = i++) {
            const ImVec2 a = points[previous];
            const ImVec2 b = points[i];
            const float dx = b.x - a.x;
            const float dy = b.y - a.y;
            const float length_squared = dx * dx + dy * dy;
            if (length_squared > 0.0f) {
                const float u = std::clamp(((pointer.x - a.x) * dx + (pointer.y - a.y) * dy) / length_squared,
                                           0.0f, 1.0f);
                const float x = pointer.x - (a.x + u * dx);
                const float y = pointer.y - (a.y + u * dy);
                if (x * x + y * y <= tolerance * tolerance) {
                    return true;
                }
            }
            if ((a.y > pointer.y) != (b.y > pointer.y) && pointer.x < a.x + (pointer.y - a.y) * dx / dy) {
                inside = !inside;
            }
        }
        return inside;
    }

    void draw(ImDrawList& draw_list, ImU32 color, bool highlighted = false) const {
        if (count < 3) {
            return;
        }
        const int point_count = static_cast<int>(count);
        const unsigned int fill = highlighted ? 30u : radius_px < 11.0f ? 50u : 81u;
        const unsigned int outline = highlighted ? 255u : radius_px < 11.0f ? 112u : 174u;
        draw_list.AddConvexPolyFilled(points.data(), point_count, footprint_detail::with_alpha(color, fill));
        draw_list.AddPolyline(points.data(), point_count, footprint_detail::with_alpha(color, outline),
                              ImDrawFlags_Closed, highlighted ? 1.8f : 0.85f);
    }

    void draw_population(ImDrawList& draw_list, ImU32 color, int population, std::size_t index, float travel) const {
        const int density = std::clamp(static_cast<int>(3.1f * std::sqrt(static_cast<float>(population))), 10, 120);
        const float limb_fade = std::clamp((facing + 0.02f) / 0.30f, 0.08f, 1.0f);
        const int grains = std::max(1, static_cast<int>(density * (1.0f - 0.82f * travel) * std::sqrt(limb_fade)));
        std::uint32_t seed = static_cast<std::uint32_t>(index + 1) * 0x9e3779b9u;
        const auto random = [&] {
            seed = seed * 1664525u + 1013904223u;
            return static_cast<float>(seed >> 8) / 16777216.0f;
        };
        ImVec2 middle{};
        for (std::size_t i = 0; i < count; ++i) {
            middle.x += points[i].x;
            middle.y += points[i].y;
        }
        middle.x /= static_cast<float>(count);
        middle.y /= static_cast<float>(count);
        draw_list.PrimReserve(grains * 6, grains * 4);
        for (int i = 0; i < grains; ++i) {
            const std::size_t edge = std::min(count - 1, static_cast<std::size_t>(random() * count));
            const ImVec2 a = points[edge];
            const ImVec2 b = points[(edge + 1) % count];
            const float along = random();
            const float outward = std::sqrt(random());
            const float x = std::lerp(middle.x, std::lerp(a.x, b.x, along), outward);
            const float y = std::lerp(middle.y, std::lerp(a.y, b.y, along), outward);
            const float half_size = 0.26f + random() * 0.39f;
            const unsigned int alpha = static_cast<unsigned int>((61.0f + random() * 152.0f) *
                (1.0f - 0.65f * travel) * limb_fade);
            const ImU32 tint = random() < 0.14f ? IM_COL32(255, 242, 220, 255) : color;
            draw_list.PrimRect({x - half_size, y - half_size}, {x + half_size, y + half_size},
                               footprint_detail::with_alpha(tint, alpha));
        }
    }
};

inline bool project_footprint(const GlobeView& globe, const Colony& colony, std::size_t index,
                              float globe_radius, ImVec2 viewport_min, ImVec2 viewport_max,
                              ProjectedFootprint& out, float min_radius_px = 5.0f) {
    using namespace footprint_detail;
    out.count = 0;
    globe_radius = std::max(1.0f, globe_radius);
    out.radius_px = std::max(colony.radius_km * globe_radius / earth_radius_km, min_radius_px);
    const float max_screen_radius = out.radius_px * maximum_extent;
    const float angular_radius = out.radius_px / globe_radius;
    float facing;
    const Vec3 origin = colony.position;
    const bool center_visible = globe.project(origin, out.center, &facing);
    out.facing = facing;
    if (facing < -angular_radius * maximum_extent ||
        (center_visible && (out.center.x + max_screen_radius < viewport_min.x ||
                            out.center.x - max_screen_radius > viewport_max.x ||
                            out.center.y + max_screen_radius < viewport_min.y ||
                            out.center.y - max_screen_radius > viewport_max.y))) {
        return false;
    }

    const RangeShape shape(index);
    const std::size_t step = out.radius_px >= 15.0f ? 1 : out.radius_px >= 5.0f ? 2 : 3;
    const float horizontal = std::hypot(origin.x, origin.z);
    const Vec3 east = horizontal > 0.0001f
        ? Vec3{-origin.z / horizontal, 0.0f, origin.x / horizontal}
        : Vec3{0.0f, 0.0f, 1.0f};
    const Vec3 north{-origin.y * east.z, origin.x * east.z - origin.z * east.x, origin.y * east.x};
    const auto vertex = [&](std::size_t i) {
        const Vec2 offset = shape.offset(i, angular_radius);
        return unit_vector({origin.x + east.x * offset.x + north.x * offset.y,
                            origin.y + north.y * offset.y,
                            origin.z + east.z * offset.x + north.z * offset.y});
    };

    Vec3 previous = vertex(range_samples - step);
    float previous_facing;
    ImVec2 unused;
    globe.project(previous, unused, &previous_facing);
    std::array<Vec3, 2> crossings;
    std::size_t crossing_count = 0;
    for (std::size_t i = 0; i < range_samples; i += step) {
        const Vec3 current = vertex(i);
        float current_facing;
        ImVec2 screen;
        const bool visible = globe.project(current, screen, &current_facing);
        if ((previous_facing < 0.0f) != (current_facing < 0.0f)) {
            const float t = previous_facing / (previous_facing - current_facing);
            const Vec3 horizon = unit_vector({std::lerp(previous.x, current.x, t),
                                             std::lerp(previous.y, current.y, t),
                                             std::lerp(previous.z, current.z, t)});
            ImVec2 edge;
            if (globe.project(horizon, edge)) {
                out.points[out.count++] = edge;
                if (crossing_count < crossings.size()) {
                    crossings[crossing_count++] = horizon;
                }
            }
        }
        if (visible && current_facing >= 0.0f) {
            out.points[out.count++] = screen;
        }
        previous = current;
        previous_facing = current_facing;
    }
    if (crossing_count == 2) {
        // The boundary at the globe's silhouette is an arc, not the chord
        // joining the two visible endpoints.
        const Vec3 midpoint = unit_vector({crossings[0].x + crossings[1].x,
                                           crossings[0].y + crossings[1].y,
                                           crossings[0].z + crossings[1].z});
        ImVec2 screen;
        if (globe.project(midpoint, screen)) {
            out.points[out.count++] = screen;
        }
    }
    if (out.count >= 3 && facing < angular_radius * maximum_extent * 2.0f) {
        out.count = limb_outline(out.points, out.count);
    }
    if (out.count < 3) {
        return false;
    }

    out.min = out.points[0];
    out.max = out.points[0];
    for (std::size_t i = 1; i < out.count; ++i) {
        const ImVec2 p = out.points[i];
        out.min.x = std::min(out.min.x, p.x);
        out.min.y = std::min(out.min.y, p.y);
        out.max.x = std::max(out.max.x, p.x);
        out.max.y = std::max(out.max.y, p.y);
    }
    if (!center_visible) {
        out.center = {(out.min.x + out.max.x) * 0.5f, (out.min.y + out.max.y) * 0.5f};
    }
    return out.max.x >= viewport_min.x && out.max.y >= viewport_min.y &&
           out.min.x <= viewport_max.x && out.min.y <= viewport_max.y;
}

} // namespace wildlife
