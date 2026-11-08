#include "globe.hpp"

#include "earth_land.hpp"
#include "earth_texture.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <numbers>
#include <optional>
#include <vector>

namespace {

using wildlife::Vec3;

constexpr std::size_t ocean_segments = 144;
constexpr std::size_t ocean_rings = 15;
constexpr std::size_t land_vertex_count = std::size(earth_land::vertices);
constexpr double radians_per_degree = std::numbers::pi / 180.0;
constexpr double degrees_per_radian = 180.0 / std::numbers::pi;
constexpr float horizon_tolerance = 1.0e-6f;
constexpr float terrain_pick_radius = 2400.0f;

struct ScreenRay {
    double east;
    double north;
    double depth;
};

std::optional<ScreenRay> screen_ray(ImVec2 screen, ImVec2 center, double radius, double offset, double scale) {
    const double east = (static_cast<double>(screen.x) - center.x) / (radius * scale);
    const double north = ((center.y - static_cast<double>(screen.y)) / radius + offset) / scale;
    const double distance_squared = east * east + north * north;
    if (distance_squared > 1.0 + 2.0 * horizon_tolerance) return std::nullopt;
    return ScreenRay{east, north, std::sqrt(std::max(0.0, 1.0 - distance_squared))};
}

struct Rgb {
    float r;
    float g;
    float b;
};

Rgb mix(Rgb a, Rgb b, float weight) {
    return {std::lerp(a.r, b.r, weight), std::lerp(a.g, b.g, weight), std::lerp(a.b, b.b, weight)};
}

ImU32 packed_color(Rgb rgb, int alpha = 255) {
    return IM_COL32(static_cast<int>(rgb.r), static_cast<int>(rgb.g), static_cast<int>(rgb.b), alpha);
}

float dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Interpolate to the view plane, then place the point on the globe's silhouette.
Vec3 horizon_point(Vec3 a, Vec3 b, float t) {
    const float x = a.x + (b.x - a.x) * t;
    const float y = a.y + (b.y - a.y) * t;
    const float inverse_radius = 1.0f / std::sqrt(x * x + y * y);
    return {x * inverse_radius, y * inverse_radius, 0.0f};
}

Vec3 horizon_point(Vec3 a, Vec3 b) {
    return horizon_point(a, b, a.z / (a.z - b.z));
}

float smooth_step(float low, float high, float value) {
    const float t = std::clamp((value - low) / (high - low), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

ImU32 surface_color(Rgb base, Vec3 normal, Vec3 light) {
    const float day = smooth_step(-0.09f, 0.15f, dot(normal, light));
    const float edge = std::max(0.0f, 1.0f - normal.z);
    const float haze = edge * edge * edge * (0.04f + 0.24f * day);
    const Rgb shaded = mix({7.0f, 12.0f, 20.0f}, base, 0.23f + 0.62f * day);
    return packed_color(mix(shaded, {90.0f, 126.0f, 157.0f}, haze));
}

struct TerrainRegion {
    float latitude;
    float longitude;
    float latitude_scale;
    float longitude_scale;
    float strength;
};

template <std::size_t N>
float region_weight(float latitude, float longitude, const TerrainRegion (&regions)[N]) {
    float weight = 0.0f;
    for (const auto& region : regions) {
        const float north = (latitude - region.latitude) / region.latitude_scale;
        const float east = std::remainder(longitude - region.longitude, 360.0f) / region.longitude_scale;
        weight = std::max(weight, region.strength * std::exp(-0.5f * (north * north + east * east)));
    }
    return weight;
}

Rgb terrain_color(Vec3 normal, float latitude, float longitude) {
    // Broad, stylized biome bands anchored in geographic space so they rotate with Earth.
    constexpr TerrainRegion dry_regions[]{
        {23, 13, 7, 21, 0.99f}, {24, 47, 7, 12, 0.9f}, {39, 86, 5, 11, 0.7f},
        {-24, 133, 8, 17, 0.92f}, {34, -111, 7, 11, 0.61f}, {-23, -71, 10, 4, 0.79f},
        {-23, 18, 9, 8, 0.64f}, {26, 70, 4, 8, 0.6f},
    };
    constexpr TerrainRegion uplands[]{
        {31, 88, 4, 11, 0.81f}, {45, 83, 6, 13, 0.48f}, {43, -114, 12, 6, 0.49f},
        {-15, -71, 16, 4, 0.67f}, {-37, -71, 12, 4, 0.54f}, {46, 11, 2, 5, 0.43f},
    };
    constexpr TerrainRegion greenland_ice[]{{73, -42, 9, 17, 1.0f}};
    const float absolute_latitude = std::abs(latitude);
    const float terrain_variation = 0.045f * std::sin(61.0f * normal.x + 29.0f * normal.z) *
                                    std::cos(43.0f * normal.y - 33.0f * normal.z);
    const float temperate = smooth_step(8.0f, 52.0f, absolute_latitude);
    const float dry = region_weight(latitude, longitude, dry_regions);
    const float upland = region_weight(latitude, longitude, uplands);
    const float arctic = smooth_step(64.0f, 86.0f, absolute_latitude);
    const float ice = std::max(arctic, latitude < 0.0f ? smooth_step(61.0f, 72.0f, -latitude) :
        smooth_step(0.32f, 0.82f, region_weight(latitude, longitude, greenland_ice)));
    Rgb color = mix({43.0f, 100.0f, 75.0f}, {101.0f, 129.0f, 85.0f},
                    std::clamp(temperate + terrain_variation, 0.0f, 1.0f));
    color = mix(color, {172.0f, 135.0f, 86.0f}, dry);
    color = mix(color, {157.0f, 147.0f, 127.0f}, upland);
    return mix(color, {214.0f, 231.0f, 228.0f}, ice);
}

const auto& unit_circle() {
    static const auto circle = [] {
        std::array<ImVec2, ocean_segments> points{};
        for (std::size_t i = 0; i < ocean_segments; ++i) {
            const float azimuth = 2.0f * std::numbers::pi_v<float> * static_cast<float>(i) / ocean_segments;
            points[i] = {std::cos(azimuth), std::sin(azimuth)};
        }
        return points;
    }();
    return circle;
}

const auto& ocean_mesh() {
    static const auto mesh = [] {
        std::array<Vec3, 1 + ocean_rings * ocean_segments> points{};
        points[0] = {0.0f, 0.0f, 1.0f};
        const auto& circle = unit_circle();
        for (std::size_t ring = 1; ring <= ocean_rings; ++ring) {
            const float angle = std::numbers::pi_v<float> * 0.5f * static_cast<float>(ring) / ocean_rings;
            const float radial = std::sin(angle);
            const float facing = std::max(0.0f, std::cos(angle));
            for (std::size_t i = 0; i < ocean_segments; ++i) {
                points[1 + (ring - 1) * ocean_segments + i] =
                    {circle[i].x * radial, circle[i].y * radial, facing};
            }
        }
        return points;
    }();
    return mesh;
}

struct LandVertex {
    Vec3 position;
    Rgb color;
};

const auto& land_mesh() {
    static const auto mesh = [] {
        std::array<LandVertex, land_vertex_count> points{};
        for (std::size_t i = 0; i < land_vertex_count; ++i) {
            const auto coordinate = earth_land::vertices[i];
            const float latitude = coordinate.latitude_centidegrees * 0.01f;
            const float longitude = coordinate.longitude_centidegrees * 0.01f;
            const Vec3 normal = wildlife::earth_position(latitude, longitude);
            points[i] = {normal, terrain_color(normal, latitude, longitude)};
        }
        return points;
    }();
    return mesh;
}

struct GridLine {
    std::vector<Vec3> points;
    ImU32 color;
};

ImU32 grid_color(int degrees) {
    if (degrees == 0) {
        return IM_COL32(169, 211, 226, 33);
    }
    return degrees % 30 == 0 ? IM_COL32(169, 211, 226, 19) : IM_COL32(169, 211, 226, 11);
}

const auto& graticule() {
    static const auto lines = [] {
        std::vector<GridLine> grid;
        for (int latitude = -75; latitude <= 75; latitude += 15) {
            GridLine& line = grid.emplace_back();
            line.color = grid_color(latitude);
            for (int longitude = -180; longitude <= 180; longitude += 3) {
                line.points.push_back(wildlife::earth_position(static_cast<float>(latitude),
                                                               static_cast<float>(longitude)));
            }
        }
        for (int longitude = -180; longitude < 180; longitude += 15) {
            GridLine& line = grid.emplace_back();
            line.color = grid_color(longitude);
            for (int latitude = -90; latitude <= 90; latitude += 3) {
                line.points.push_back(wildlife::earth_position(static_cast<float>(latitude),
                                                               static_cast<float>(longitude)));
            }
        }
        return grid;
    }();
    return lines;
}

} // namespace

GlobeView::GlobeView(ImVec2 center, float radius, double latitude_degrees, double longitude_degrees, float tilt_degrees)
    : center_(center), radius_(std::max(radius, 1.0f)) {
    const double latitude = latitude_degrees * radians_per_degree;
    const double longitude = longitude_degrees * radians_per_degree;
    latitude_ = latitude;
    longitude_ = longitude;
    const float sin_latitude = static_cast<float>(std::sin(latitude));
    const float cos_latitude = static_cast<float>(std::cos(latitude));
    const float sin_longitude = static_cast<float>(std::sin(longitude));
    const float cos_longitude = static_cast<float>(std::cos(longitude));
    east_ = {-sin_longitude, 0.0f, cos_longitude};
    north_ = {-sin_latitude * cos_longitude, cos_latitude, -sin_latitude * sin_longitude};
    forward_ = {cos_latitude * cos_longitude, sin_latitude, cos_latitude * sin_longitude};
    view_up_ = north_;
    view_front_ = forward_;
    if (tilt_degrees != 0.0f && wildlife::earth_terrain_available()) {
        const double tilt = tilt_degrees * radians_per_degree;
        tilt_cos_ = std::cos(tilt);
        tilt_sin_ = std::sin(tilt);
        const auto c = static_cast<float>(tilt_cos_), s = static_cast<float>(tilt_sin_);
        view_up_ = {north_.x * c + forward_.x * s, north_.y * c + forward_.y * s, north_.z * c + forward_.z * s};
        view_front_ = {forward_.x * c - north_.x * s, forward_.y * c - north_.y * s, forward_.z * c - north_.z * s};
        pivot_radius_ += wildlife::earth_surface_height(forward_) / wildlife::earth_radius_meters;
        pivot_y_ = pivot_radius_ * tilt_sin_;
    }
}

Vec3 GlobeView::view_space(Vec3 point) const {
    return {dot(point, east_), dot(point, view_up_), dot(point, view_front_)};
}

ImVec2 GlobeView::screen_position(Vec3 view_point) const {
    return {center_.x + view_point.x * radius_, center_.y - view_point.y * radius_};
}

bool GlobeView::project(double latitude_degrees, double longitude_degrees,
                        ImVec2& screen, float* facing) const {
    if (radius_ / wildlife::earth_radius_meters >= .08) {
        const double lat = latitude_degrees * radians_per_degree, lon = longitude_degrees * radians_per_degree;
        const double dlon = lon - longitude_;
        const double east = std::cos(lat) * std::sin(dlon) * wildlife::earth_radius_meters;
        const double north = (std::sin(lat)*std::cos(latitude_) - std::cos(lat)*std::sin(latitude_)*std::cos(dlon)) * wildlife::earth_radius_meters;
        const double front = std::sin(lat)*std::sin(latitude_) + std::cos(lat)*std::cos(latitude_)*std::cos(dlon);
        if (facing) *facing = static_cast<float>(front);
        if (front < 0) return false;
        const double height = wildlife::sample_local_terrain(lat,lon).height - wildlife::sample_local_terrain(latitude_,longitude_).height
            - (east*east+north*north)/(2*wildlife::earth_radius_meters);
        const double ppm = radius_/wildlife::earth_radius_meters;
        screen = {static_cast<float>(center_.x + east*ppm), static_cast<float>(center_.y-(north*tilt_cos_+height*tilt_sin_)*ppm)};
        return true;
    }
    return project(wildlife::earth_position(static_cast<float>(latitude_degrees),
                                            static_cast<float>(longitude_degrees)),
                   screen, facing);
}

bool GlobeView::project(float x, float y, float z, ImVec2& screen, float* facing) const {
    return project(Vec3{x, y, z}, screen, facing);
}

bool GlobeView::project(Vec3 point, ImVec2& screen, float* facing) const {
    if (radius_ / wildlife::earth_radius_meters >= .08) {
        return project(std::atan2(double(point.y),std::hypot(double(point.x),double(point.z)))*degrees_per_radian,
                       std::atan2(double(point.z),double(point.x))*degrees_per_radian, screen, facing);
    }
    const auto view = view_space(point);
    if (facing) {
        *facing = std::clamp(view.z, -1.0f, 1.0f);
    }
    if (view.z < -horizon_tolerance) {
        return false;
    }
    const double scale = radius_ >= terrain_pick_radius || tilt_sin_ != 0.0
        ? 1.0 + wildlife::earth_surface_height(point) / wildlife::earth_radius_meters : 1.0;
    screen = {static_cast<float>(center_.x + view.x * scale * radius_),
              static_cast<float>(center_.y - (view.y * scale - pivot_y_) * radius_)};
    return true;
}

float GlobeView::edge_clearance(ImVec2 screen) const {
    return radius_ - static_cast<float>(std::hypot(screen.x - center_.x, screen.y - center_.y - pivot_y_ * radius_));
}

bool GlobeView::unproject(ImVec2 screen, double& latitude_degrees, double& longitude_degrees) const {
    if (radius_ / wildlife::earth_radius_meters >= .08) {
        const double ppm=radius_/wildlife::earth_radius_meters;
        const double east=(screen.x-center_.x)/ppm, y=(center_.y-screen.y)/ppm;
        double north=y/tilt_cos_;
        const double datum=wildlife::sample_local_terrain(latitude_,longitude_).height;
        const auto location = [&](double n) {
            const double e=east/wildlife::earth_radius_meters, z=n/wildlife::earth_radius_meters;
            const double x=std::cos(latitude_)*std::cos(longitude_)-e*std::sin(longitude_)-z*std::sin(latitude_)*std::cos(longitude_);
            const double a=std::sin(latitude_)+z*std::cos(latitude_);
            const double b=std::cos(latitude_)*std::sin(longitude_)+e*std::cos(longitude_)-z*std::sin(latitude_)*std::sin(longitude_);
            return std::array{std::atan2(a,std::hypot(x,b)),std::atan2(b,x)};
        };
        for(int i=0;i<16;++i){
            const auto p=location(north);
            const double h=wildlife::sample_local_terrain(p[0],p[1]).height-datum-(east*east+north*north)/(2*wildlife::earth_radius_meters);
            const double next=(y-h*tilt_sin_)/tilt_cos_;
            if(std::abs(next-north)<.001){north=next;break;}
            north=std::lerp(north,next,.5);
        }
        const auto p=location(north);
        latitude_degrees=p[0]*degrees_per_radian;longitude_degrees=p[1]*degrees_per_radian;return true;
    }
    const auto ray = screen_ray(screen, center_, radius_, pivot_y_, 1.0);
    if (!ray) return false;
    const auto point_at = [&](double depth) {
        return std::array{ray->east * east_.x + ray->north * view_up_.x + depth * view_front_.x,
                          ray->north * view_up_.y + depth * view_front_.y,
                          ray->east * east_.z + ray->north * view_up_.z + depth * view_front_.z};
    };
    double depth = ray->depth;
    if (wildlife::earth_terrain_available() && (radius_ >= terrain_pick_radius || tilt_sin_ != 0.0)) {
        // Walk the ray from the highest possible surface toward sea level, then refine the first hit.
        constexpr double limit = 1.0 + wildlife::max_land_elevation / wildlife::earth_radius_meters;
        const double near = std::sqrt(depth * depth + limit * limit - 1.0);
        const auto below_surface = [&](double distance) {
            const auto [x, y, z] = point_at(distance);
            const auto height = wildlife::earth_surface_height({static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)});
            const auto ray_height = (std::sqrt(ray->east * ray->east + ray->north * ray->north + distance * distance) - 1.0)
                                    * wildlife::earth_radius_meters;
            return ray_height <= height;
        };
        double previous = near;
        constexpr int steps = 64;
        for (int step = 1; step <= steps; ++step) {
            const double next = std::lerp(near, ray->depth, static_cast<double>(step) / steps);
            if (below_surface(next)) {
                double low = next, high = previous;
                for (int i = 0; i < 14; ++i) {
                    const double middle = (low + high) * 0.5;
                    if (below_surface(middle)) low = middle;
                    else high = middle;
                }
                depth = (low + high) * 0.5;
                break;
            }
            previous = next;
        }
    }
    const auto [x, y, z] = point_at(depth);
    latitude_degrees = std::atan2(y, std::hypot(x, z)) * degrees_per_radian;
    longitude_degrees = std::atan2(z, x) * degrees_per_radian;
    return true;
}

bool GlobeView::center_for_anchor(ImVec2 screen, double latitude, double longitude,
                                  double& center_latitude, double& center_longitude) const {
    if (radius_ / wildlife::earth_radius_meters >= .08) {
        const double ppm=radius_/wildlife::earth_radius_meters;
        const double east=(screen.x-center_.x)/ppm, y=(center_.y-screen.y)/ppm;
        const double h=wildlife::sample_local_terrain(latitude*radians_per_degree,longitude*radians_per_degree).height;
        double lat=latitude_,lon=longitude_;
        for(int i=0;i<8;++i){
            const double datum=wildlife::sample_local_terrain(lat,lon).height;
            const double north=(y-(h-datum)*tilt_sin_)/tilt_cos_;
            lat=latitude*radians_per_degree-north/wildlife::earth_radius_meters;
            lon=longitude*radians_per_degree-east/(wildlife::earth_radius_meters*std::max(.001,std::cos(lat)));
        }
        center_latitude=std::clamp(lat*degrees_per_radian,-89.999999,89.999999);
        center_longitude=std::remainder(lon*degrees_per_radian,360.0);return true;
    }
    constexpr double epsilon = 1.0e-10;
    constexpr double half_turn = std::numbers::pi;
    constexpr double pole = half_turn * 0.5;
    const double target_y = std::sin(latitude * radians_per_degree);
    const auto point = wildlife::earth_position(static_cast<float>(latitude), static_cast<float>(longitude));
    const double scale = radius_ >= terrain_pick_radius || tilt_sin_ != 0.0
        ? 1.0 + wildlife::earth_surface_height(point) / wildlife::earth_radius_meters : 1.0;
    const double current_yaw = std::atan2(-east_.x, east_.z);
    double pivot = pivot_y_, result_lat = 0.0, result_lon = 0.0;
    for (int iteration = 0; iteration < 8; ++iteration) {
        const auto ray = screen_ray(screen, center_, radius_, pivot, scale);
        if (!ray) return false;
        const double north = ray->north * tilt_cos_ - ray->depth * tilt_sin_;
        const double forward = ray->north * tilt_sin_ + ray->depth * tilt_cos_;
        const double vertical = std::hypot(north, forward);
        if (vertical < epsilon || std::abs(target_y) > vertical + epsilon) return false;

        // World Y is north * cos(pitch) + forward * sin(pitch).
        const double angle = std::asin(std::clamp(target_y / vertical, -1.0, 1.0));
        const double offset = std::atan2(north, forward);
        double best = -4.0;
        for (const double root : {angle, half_turn - angle}) {
            const double pitch = std::remainder(root - offset, 2.0 * half_turn);
            if (std::abs(pitch) > pole + epsilon) continue;
            const double sine = std::sin(pitch);
            const double cosine = std::cos(pitch);
            const double along = forward * cosine - north * sine;
            const double yaw = std::hypot(ray->east, along) < epsilon ? current_yaw
                : longitude * radians_per_degree - std::atan2(ray->east, along);
            const double yaw_alignment = std::cos(yaw - current_yaw);
            const double pitch_alignment = forward_.y * sine + north_.y * cosine;
            const double alignment = yaw_alignment + (1.0 + yaw_alignment) * pitch_alignment;
            if (alignment <= best) continue;
            best = alignment;
            result_lat = std::clamp(pitch * degrees_per_radian, -90.0, 90.0);
            result_lon = std::remainder(yaw * degrees_per_radian, 360.0);
        }
        if (best == -4.0) return false;
        if (tilt_sin_ == 0.0) break;
        const auto new_center = wildlife::earth_position(static_cast<float>(result_lat), static_cast<float>(result_lon));
        const double new_pivot = tilt_sin_ * (1.0 + wildlife::earth_surface_height(new_center) / wildlife::earth_radius_meters);
        if (std::abs(new_pivot - pivot) * radius_ < 0.03) break;
        pivot = new_pivot;
    }
    center_latitude = result_lat;
    center_longitude = result_lon;
    return true;
}

void GlobeView::draw(ImDrawList& draw_list, Vec3 light) const {
    const Vec3 view_light{light.x, static_cast<float>(light.y * tilt_cos_ + light.z * tilt_sin_),
                         static_cast<float>(light.z * tilt_cos_ - light.y * tilt_sin_)};
    if (wildlife::draw_earth_texture(draw_list, center_, radius_, east_, view_up_, view_front_, view_light,
                                     static_cast<float>(pivot_y_), static_cast<float>(pivot_radius_))) {
        return;
    }
    draw_atmosphere(draw_list, light);
    draw_ocean(draw_list, light);
    draw_land(draw_list, light);
    draw_graticule(draw_list);
}

void GlobeView::draw_atmosphere(ImDrawList& draw_list, Vec3 light) const {
    const float spread = std::clamp(radius_ * 0.065f, 9.0f, 32.0f);
    constexpr std::array stops{0.0f, 0.055f, 0.14f, 0.3f, 0.55f, 1.0f};
    constexpr std::array opacity{167.0f, 98.0f, 57.0f, 28.0f, 9.0f, 0.0f};
    constexpr int index_count = 6 * ocean_segments * (stops.size() - 1);
    constexpr int vertex_count = ocean_segments * stops.size();
    const auto& circle = unit_circle();
    draw_list.PrimReserve(index_count, vertex_count);
    const unsigned int base_index = draw_list._VtxCurrentIdx;
    const ImVec2 white = ImGui::GetFontTexUvWhitePixel();

    for (std::size_t ring = 0; ring < stops.size(); ++ring) {
        const std::size_t base = base_index + ring * ocean_segments;
        const float radius = radius_ + spread * stops[ring];
        for (std::size_t i = 0; i < ocean_segments; ++i) {
            const ImVec2 direction = circle[i];
            const float day = smooth_step(-0.20f, 0.68f, direction.x * light.x + direction.y * light.y);
            const float intensity = 0.18f + 0.82f * day;
            const Rgb blue = mix({46.0f, 66.0f, 97.0f}, {171.0f, 196.0f, 214.0f}, day);
            draw_list.PrimWriteVtx({center_.x + radius * direction.x, center_.y - radius * direction.y},
                                   white, packed_color(blue, static_cast<int>(opacity[ring] * intensity)));
            if (ring + 1 < stops.size()) {
                const std::size_t left = base + i;
                const std::size_t right = base + (i + 1) % ocean_segments;
                draw_list.PrimWriteIdx(static_cast<ImDrawIdx>(left));
                draw_list.PrimWriteIdx(static_cast<ImDrawIdx>(left + ocean_segments));
                draw_list.PrimWriteIdx(static_cast<ImDrawIdx>(right));
                draw_list.PrimWriteIdx(static_cast<ImDrawIdx>(right));
                draw_list.PrimWriteIdx(static_cast<ImDrawIdx>(left + ocean_segments));
                draw_list.PrimWriteIdx(static_cast<ImDrawIdx>(right + ocean_segments));
            }
        }
    }
}

void GlobeView::draw_ocean(ImDrawList& draw_list, Vec3 light) const {
    const auto& mesh = ocean_mesh();
    constexpr int index_count = 3 * ocean_segments * (1 + 2 * (ocean_rings - 1));
    draw_list.PrimReserve(index_count, static_cast<int>(mesh.size()));
    const unsigned int base_index = draw_list._VtxCurrentIdx;
    const ImVec2 white = ImGui::GetFontTexUvWhitePixel();

    auto triangle = [&](std::size_t a, std::size_t b, std::size_t c) {
        draw_list.PrimWriteIdx(static_cast<ImDrawIdx>(base_index + a));
        draw_list.PrimWriteIdx(static_cast<ImDrawIdx>(base_index + b));
        draw_list.PrimWriteIdx(static_cast<ImDrawIdx>(base_index + c));
    };
    for (std::size_t i = 0; i < ocean_segments; ++i) {
        triangle(0, 1 + i, 1 + (i + 1) % ocean_segments);
    }
    for (std::size_t ring = 1; ring < ocean_rings; ++ring) {
        const std::size_t inner_start = 1 + (ring - 1) * ocean_segments;
        const std::size_t outer_start = inner_start + ocean_segments;
        for (std::size_t i = 0; i < ocean_segments; ++i) {
            const std::size_t next = (i + 1) % ocean_segments;
            triangle(inner_start + i, outer_start + i, inner_start + next);
            triangle(inner_start + next, outer_start + i, outer_start + next);
        }
    }
    for (Vec3 point : mesh) {
        constexpr Vec3 reflection{-0.249f, 0.292f, 0.923f};
        const float shine = smooth_step(0.9f, 1.0f, dot(point, reflection));
        const float world_y = point.y * north_.y + point.z * forward_.y;
        const float polar_ice = smooth_step(0.976f, 0.999f, world_y) * 0.45f;
        Rgb color = mix({24.0f, 49.0f, 72.0f}, {47.0f, 80.0f, 98.0f}, shine * 0.7f);
        color = mix(color, {160.0f, 192.0f, 198.0f}, polar_ice);
        draw_list.PrimWriteVtx(screen_position(point), white, surface_color(color, point, light));
    }
}

void GlobeView::draw_land(ImDrawList& draw_list, Vec3 light) const {
    struct MeshScratch {
        std::array<Vec3, land_vertex_count> projected;
        std::vector<LandVertex> intersections;
        std::vector<std::size_t> indices;
    };
    thread_local MeshScratch scratch;
    scratch.intersections.clear();
    scratch.indices.clear();
    const auto& mesh = land_mesh();
    for (std::size_t i = 0; i < land_vertex_count; ++i) {
        scratch.projected[i] = view_space(mesh[i].position);
    }
    for (const auto& triangle : earth_land::triangles) {
        std::array<std::size_t, 4> visible{};
        std::size_t count = 0;
        std::size_t previous_index = triangle[2];
        for (std::size_t index : triangle) {
            const Vec3 previous = scratch.projected[previous_index];
            const Vec3 current = scratch.projected[index];
            if ((previous.z < 0.0f) != (current.z < 0.0f)) {
                const float fraction = previous.z / (previous.z - current.z);
                visible[count++] = land_vertex_count + scratch.intersections.size();
                scratch.intersections.push_back({horizon_point(previous, current, fraction),
                    mix(mesh[previous_index].color, mesh[index].color, fraction)});
            }
            if (current.z >= 0.0f) {
                visible[count++] = index;
            }
            previous_index = index;
        }
        for (std::size_t i = 2; i < count; ++i) {
            scratch.indices.insert(scratch.indices.end(), {visible[0], visible[i - 1], visible[i]});
        }
    }

    draw_list.PrimReserve(static_cast<int>(scratch.indices.size()),
                          static_cast<int>(land_vertex_count + scratch.intersections.size()));
    const unsigned int base_index = draw_list._VtxCurrentIdx;
    const ImVec2 white = ImGui::GetFontTexUvWhitePixel();
    for (std::size_t index : scratch.indices) {
        draw_list.PrimWriteIdx(static_cast<ImDrawIdx>(base_index + index));
    }
    for (std::size_t i = 0; i < land_vertex_count; ++i) {
        const Vec3 point = scratch.projected[i];
        draw_list.PrimWriteVtx(screen_position(point), white, surface_color(mesh[i].color, point, light));
    }
    for (const auto& vertex : scratch.intersections) {
        draw_list.PrimWriteVtx(screen_position(vertex.position), white, surface_color(vertex.color, vertex.position, light));
    }
}

void GlobeView::draw_graticule(ImDrawList& draw_list) const {
    for (const GridLine& line : graticule()) {
        std::array<ImVec2, 122> path;
        std::size_t count = 0;
        Vec3 previous = view_space(line.points.front());
        if (previous.z >= 0.0f) {
            path[count++] = screen_position(previous);
        }
        for (std::size_t i = 1; i < line.points.size(); ++i) {
            const Vec3 current = view_space(line.points[i]);
            if ((previous.z < 0.0f) != (current.z < 0.0f)) {
                path[count++] = screen_position(horizon_point(previous, current));
            }
            if (current.z >= 0.0f) {
                path[count++] = screen_position(current);
            } else if (count > 0) {
                draw_list.AddPolyline(path.data(), static_cast<int>(count), line.color, ImDrawFlags_None, 1.0f);
                count = 0;
            }
            previous = current;
        }
        if (count > 0) {
            draw_list.AddPolyline(path.data(), static_cast<int>(count), line.color, ImDrawFlags_None, 1.0f);
        }
    }
}
