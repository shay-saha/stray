#pragma once

#include "simulation.hpp"

#include <imgui.h>

// North-up orthographic camera with an optional tilt around the map center.
class GlobeView {
public:
    static constexpr wildlife::Vec3 sunlight{-0.86f, 0.37f, 0.35f};

    GlobeView(ImVec2 center, float radius, double latitude_degrees, double longitude_degrees, float tilt_degrees = 0.0f);

    bool project(double latitude_degrees, double longitude_degrees,
                 ImVec2& screen, float* facing = nullptr) const;
    bool project(float x, float y, float z, ImVec2& screen, float* facing = nullptr) const;
    bool project(wildlife::Vec3 point, ImVec2& screen, float* facing = nullptr) const;
    float edge_clearance(ImVec2 screen) const;
    bool unproject(ImVec2 screen, double& latitude_degrees, double& longitude_degrees) const;
    // Find the north-up camera center that projects this location onto screen at this radius.
    bool center_for_anchor(ImVec2 screen, double latitude, double longitude,
                           double& center_latitude, double& center_longitude) const;
    void draw(ImDrawList& draw_list, wildlife::Vec3 light = sunlight) const;

private:
    wildlife::Vec3 view_space(wildlife::Vec3 point) const;
    ImVec2 screen_position(wildlife::Vec3 view_point) const;
    void draw_atmosphere(ImDrawList& draw_list, wildlife::Vec3 light) const;
    void draw_ocean(ImDrawList& draw_list, wildlife::Vec3 light) const;
    void draw_land(ImDrawList& draw_list, wildlife::Vec3 light) const;
    void draw_graticule(ImDrawList& draw_list) const;

    ImVec2 center_;
    float radius_;
    wildlife::Vec3 east_;
    wildlife::Vec3 north_;
    wildlife::Vec3 forward_;
    wildlife::Vec3 view_up_;
    wildlife::Vec3 view_front_;
    double tilt_cos_ = 1.0;
    double tilt_sin_ = 0.0;
    double pivot_radius_ = 1.0;
    double pivot_y_ = 0.0;
    double latitude_ = 0.0, longitude_ = 0.0;
};
