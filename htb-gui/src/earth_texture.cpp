#include "earth_texture.hpp"
#include "earth_tiles.hpp"

#include <imgui_impl_opengl3_loader.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_NO_STDIO
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <memory>
#include <numbers>
#include <vector>

namespace wildlife {
namespace {

constexpr int globe_columns = 208, globe_rows = 112;
constexpr int detail_columns = 320, detail_rows = 256;
constexpr int globe_index_count = globe_columns * globe_rows * 6;
constexpr int detail_index_count = detail_columns * detail_rows * 6;
constexpr GLenum depth_buffer_bit = 0x0100;

auto asset_paths(const std::filesystem::path& executable, const char* filename) {
    const auto bin = executable.parent_path();
    return std::array{bin / filename, bin.parent_path() / "Resources" / filename,
                      std::filesystem::path("assets") / filename};
}

struct TerrainGpu {
    struct ColorMap {
        int width = 0;
        int height = 0;
        std::vector<unsigned char> pixels;
    };
    GLuint program = 0;
    GLuint texture = 0;
    GLuint bathymetry = 0;
    GLuint cities = 0;
    GLuint elevation = 0;
    GLuint vao = 0;
    GLuint indices = 0;
    GLint camera = -1;
    GLint layout = -1;
    GLint mesh_mode = -1;
    int elevation_width = 0;
    int elevation_height = 0;
    std::unique_ptr<stbi_us, decltype(&stbi_image_free)> meters{nullptr, stbi_image_free};
    ColorMap satellite_samples;
    ColorMap ocean_samples;
    EarthTileView detail;
    std::unique_ptr<EarthTiles> tiles;
};

TerrainGpu terrain;

constexpr const char* shader_common = R"glsl(
uniform mat4 camera;
uniform mat4 layoutData;
uniform int meshMode;
uniform sampler2D elevationTiles;
uniform sampler2D tilePages;
const float pi = 3.141592653589793;
const float tileSize = 256.0;
const float atlasSize = 2560.0;
const float mercatorLimit = 0.9962720762;

vec2 globe_uv(vec3 direction) {
    float horizontal = length(direction.xz);
    return vec2(horizontal > 0.00001 ? atan(direction.z, direction.x) / (2.0 * pi) + 0.5 : 0.5,
                0.5 - atan(direction.y, horizontal) / pi);
}

vec2 tile_coord(float u, float sinLatitude) {
    vec3 window = layoutData[3].xyz;
    float v = 0.5 - atanh(clamp(sinLatitude, -mercatorLimit, mercatorLimit)) / (2.0 * pi);
    vec2 tile = vec2(u, v) * window.z;
    tile.x += window.z * round((window.x + 5.0 - tile.x) / window.z);
    return tile;
}

ivec4 page_at(ivec2 cell) {
    if (any(lessThan(cell, ivec2(0))) || any(greaterThanEqual(cell, ivec2(10)))) return ivec4(0);
    return ivec4(texelFetch(tilePages, cell, 0) * 255.0 + 0.5);
}

vec2 page_pixel(vec2 tile, ivec2 cell, ivec4 page) {
    float scale = exp2(float(page.z));
    return (tile / scale - floor((layoutData[3].xy + vec2(cell)) / scale)) * tileSize;
}

vec2 atlas_uv(ivec4 page, vec2 pixel) {
    vec2 origin = vec2(page.xy - 1) * tileSize;
    return (origin + clamp(pixel, vec2(0.5), vec2(tileSize - 0.5))) / atlasSize;
}

float elevation_meters(ivec4 page, vec2 pixel) {
    vec3 packedHeight = textureLod(elevationTiles, atlas_uv(page, pixel), 0.0).rgb * 255.0;
    return dot(packedHeight, vec3(256.0, 1.0, 1.0 / 256.0)) - 32768.0;
}
)glsl";

constexpr const char* vertex_source = R"glsl(
uniform sampler2D globeElevation;
out vec3 worldDirection;
out vec3 viewPosition;
out vec2 haloPlane;

float page_height(vec2 tile, ivec2 cell, float fallback) {
    ivec4 page = page_at(cell);
    if (page.x == 0 || (page.w & 2) == 0) return fallback;
    float opacity = float(page.w >> 2) / 63.0;
    float meters = elevation_meters(page, page_pixel(tile, cell, page));
    return mix(fallback, clamp(meters, 0.0, 9000.0), opacity);
}

float terrain_height(vec3 direction) {
    vec2 uv = globe_uv(direction);
    float fallback = textureLod(globeElevation, uv, 0.0).r * 65535.0;
    if (layoutData[3].z <= 0.0 || abs(direction.y) >= mercatorLimit) return fallback;
    vec2 tile = tile_coord(uv.x, direction.y);
    ivec2 cell = ivec2(floor(tile - layoutData[3].xy));
    vec2 fraction = fract(tile);
    vec2 weight = 0.5 * (1.0 - smoothstep(vec2(0.0), vec2(10.0 / tileSize),
                                         min(fraction, 1.0 - fraction)));
    ivec2 adjacent = ivec2(fraction.x < 0.5 ? -1 : 1, fraction.y < 0.5 ? -1 : 1);
    float height = page_height(tile, cell, fallback);
    if (weight.x > 0.0) {
        height = mix(height, page_height(tile, cell + ivec2(adjacent.x, 0), fallback), weight.x);
    }
    if (weight.y > 0.0) {
        ivec2 nextRow = cell + ivec2(0, adjacent.y);
        float other = page_height(tile, nextRow, fallback);
        if (weight.x > 0.0) {
            other = mix(other, page_height(tile, nextRow + ivec2(adjacent.x, 0), fallback), weight.x);
        }
        height = mix(height, other, weight.y);
    }
    return height;
}

void main() {
    vec2 pixel;
    float depth = 0.0;
    if (meshMode == 0) {
        vec2 corner = vec2(gl_VertexID == 1 || gl_VertexID == 2 ? 1.0 : -1.0,
                           gl_VertexID >= 2 ? 1.0 : -1.0);
        haloPlane = 1.16 * corner;
        pixel = camera[3].xy + (haloPlane - vec2(0.0, layoutData[2].w)) * vec2(1.0, -1.0) * camera[3].z;
    } else {
        // The two index grids are defined by the matching dimensions in C++.
        ivec2 grid = meshMode == 1 ? ivec2(208, 112) : ivec2(320, 256);
        ivec2 index = ivec2(gl_VertexID % (grid.x + 1), gl_VertexID / (grid.x + 1));
        vec2 uv = vec2(index) / vec2(grid);
        vec3 viewDirection;
        if (meshMode == 1) {
            float bearing = (index.x == grid.x ? 0.0 : uv.x) * 2.0 * pi;
            // Include the far side around the limb so elevated peaks can break the horizon.
            float polar = uv.y * (pi * 0.5 + 0.075);
            viewDirection = vec3(vec2(cos(bearing), sin(bearing)) * sin(polar), cos(polar));
        } else {
            vec2 plane = mix(layoutData[0].xy, layoutData[0].zw, uv);
            viewDirection = vec3(plane, sqrt(max(0.0, 1.0 - dot(plane, plane))));
        }
        worldDirection = camera[0].xyz * viewDirection.x + camera[1].xyz * viewDirection.y +
                         camera[2].xyz * viewDirection.z;
        float radialOffset = terrain_height(worldDirection) / 6371000.0;
        viewPosition = viewDirection * (1.0 + radialOffset);
        vec2 position = viewDirection.xy - vec2(0.0, layoutData[2].w) + viewDirection.xy * radialOffset;
        pixel = camera[3].xy + position * vec2(1.0, -1.0) * camera[3].z;
        // Larger view-space depth is closer to the camera.
        depth = -viewPosition.z * 0.98;
    }
    vec2 viewport = (pixel - layoutData[1].xy) / layoutData[1].zw;
    gl_Position = vec4(viewport * vec2(2.0, -2.0) + vec2(-1.0, 1.0), depth, 1.0);
}
)glsl";

constexpr const char* fragment_source = R"glsl(
uniform sampler2D atlas;
uniform sampler2D oceanRelief;
uniform sampler2D cityLights;
uniform sampler2D satelliteTiles;
in vec3 worldDirection;
in vec3 viewPosition;
in vec2 haloPlane;
out vec4 fragment;

void sample_detail(vec2 uv, vec3 normal, vec3 sun, inout vec3 photo, inout float land, inout float incidence) {
    if (layoutData[3].z <= 0.0 || abs(normal.y) >= mercatorLimit) return;
    vec2 tile = tile_coord(uv.x, normal.y);
    ivec2 cell = ivec2(floor(tile - layoutData[3].xy));
    ivec4 page = page_at(cell);
    if (page.x == 0) return;
    float count = layoutData[3].z * exp2(-float(page.z));
    vec2 pixel = page_pixel(tile, cell, page);
    float fade = float(page.w >> 2) / 63.0;
    vec3 satellite = (page.w & 1) != 0 ? texture(satelliteTiles, atlas_uv(page, pixel)).rgb : photo;

    if ((page.w & 2) != 0) {
        float height = elevation_meters(page, pixel);
        float lowland = smoothstep(-0.35, 9.0, height);
        float shore = smoothstep(0.16, 0.30, satellite.r) * smoothstep(-0.08, 0.0, satellite.r - satellite.b);
        shore *= smoothstep(-4.0, 0.25, height);
        land = mix(land, max(lowland, shore), fade);
        float horizontal = length(normal.xz);
        float metersPerPixel = 40075017.0 * horizontal / (count * tileSize);
        float eastSlope = (elevation_meters(page, pixel + vec2(1.0, 0.0)) -
                           elevation_meters(page, pixel - vec2(1.0, 0.0))) / (2.0 * metersPerPixel);
        float northSlope = (elevation_meters(page, pixel - vec2(0.0, 1.0)) -
                            elevation_meters(page, pixel + vec2(0.0, 1.0))) / (2.0 * metersPerPixel);
        vec3 east = vec3(-normal.z, 0.0, normal.x) / horizontal;
        vec3 north = cross(east, normal);
        vec3 reliefNormal = normalize(normal - 0.68 * (eastSlope * east + northSlope * north));
        vec3 worldSun = camera[0].xyz * sun.x + camera[1].xyz * sun.y + camera[2].xyz * sun.z;
        incidence = mix(incidence, dot(reliefNormal, worldSun), 0.72 * fade * land);
    }
    photo = mix(photo, satellite, fade);
}

vec3 rim_color(float light) {
    return mix(vec3(0.18, 0.26, 0.38), vec3(0.67, 0.77, 0.84), light);
}

void main() {
    vec3 sun = layoutData[2].xyz;
    if (meshMode == 0) {
        float distanceFromCenter = length(haloPlane);
        float altitude = max(0.0, distanceFromCenter - 1.0);
        float rimLight = smoothstep(-0.20, 0.68, dot(haloPlane / max(distanceFromCenter, 0.0001), normalize(sun.xy)));
        float falloff = 0.52 * exp(-76.0 * altitude) + 0.28 * exp(-20.0 * altitude);
        float air = falloff * (0.18 + 0.82 * rimLight) * (1.0 - smoothstep(0.075, 0.16, altitude));
        fragment = vec4(rim_color(rimLight), air);
        return;
    }

    vec3 normal = normalize(worldDirection);
    vec3 viewNormal = normalize(viewPosition);
    vec2 uv = globe_uv(normal);
    // Unwrap horizontal derivatives so the antimeridian does not pick a blurred mip level.
    vec2 dx = dFdx(uv);
    vec2 dy = dFdy(uv);
    dx.x -= round(dx.x);
    dy.x -= round(dy.x);
    vec3 source = textureGrad(atlas, uv, dx, dy).rgb;
    vec3 bathy = textureGrad(oceanRelief, uv, dx, dy).rgb;
    float land = smoothstep(0.035, 0.085, max(source.r, source.g));
    land *= 1.0 - smoothstep(0.48, 0.94, bathy.b) * (1.0 - smoothstep(0.12, 0.26, source.r));
    float incidence = dot(viewNormal, sun);
    // Screen derivatives give the normal of the displaced triangle that is visible here.
    vec3 triangleNormal = normalize(cross(dFdx(viewPosition), dFdy(viewPosition)));
    if (dot(triangleNormal, viewNormal) < 0.0) triangleNormal = -triangleNormal;
    float reliefIncidence = mix(incidence, dot(triangleNormal, sun), 0.64);
    sample_detail(uv, normal, sun, source, land, reliefIncidence);
    float luminance = dot(source, vec3(0.2126, 0.7152, 0.0722));
    vec3 neutral = mix(vec3(luminance), source, 0.57);
    vec3 rock = max(vec3(0.0), sqrt(neutral + 0.003) * vec3(0.47, 0.48, 0.50) - 0.036);
    vec3 water = mix(vec3(0.062, 0.154, 0.193), vec3(0.037, 0.110, 0.157),
                     smoothstep(0.001, 0.20, bathy.r));
    water = mix(water, vec3(0.016, 0.055, 0.094), smoothstep(0.18, 0.69, bathy.r));
    water = mix(water, vec3(0.008, 0.016, 0.030), smoothstep(0.68, 1.0, bathy.r));
    float relief = bathy.g - 0.502;
    water = max(vec3(0.0), water * (1.0 + 0.90 * relief) + relief * vec3(0.035, 0.066, 0.095));

    float day = smoothstep(-0.09, 0.15, incidence);
    float diffuse = sqrt(max(0.0, mix(incidence, reliefIncidence, land)));
    vec3 daylight = vec3(0.71, 0.71, 0.73) + diffuse * vec3(0.29, 0.25, 0.20);
    vec3 light = mix(vec3(0.22, 0.27, 0.36), daylight, day);
    float twilight = 1.0 - smoothstep(0.0, 0.16, abs(incidence));
    vec3 color = mix(water, rock, land) * light + twilight * vec3(0.017, 0.011, 0.007);
    float night = 1.0 - smoothstep(-0.16, 0.04, incidence);
    float emission = textureGrad(cityLights, uv, dx, dy).r;
    float lights = pow(smoothstep(0.055, 0.96, emission), 1.22) * night;
    color += vec3(0.67, 0.45, 0.25) * lights;
    vec2 outward = viewNormal.xy / max(length(viewNormal.xy), 0.0001);
    float rimLight = smoothstep(-0.20, 0.68, dot(outward, normalize(sun.xy)));
    float limb = exp(-13.0 * max(0.0, viewNormal.z));
    color = mix(color, rim_color(rimLight), limb * (0.15 + 0.58 * rimLight));
    fragment = vec4(color, 1.0);
}
)glsl";

GLuint compile_shader(GLenum kind, const char* source) {
    const GLuint shader = glCreateShader(kind);
    const char* parts[]{"#version 150\n", shader_common, source};
    glShaderSource(shader, 3, parts, nullptr);
    glCompileShader(shader);
    GLint success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (success) {
        return shader;
    }
    char message[1024]{};
    glGetShaderInfoLog(shader, sizeof(message), nullptr, message);
    std::fprintf(stderr, "Terrain shader: %s\n", message);
    glDeleteShader(shader);
    return 0;
}

GLuint terrain_program() {
    const GLuint vertex = compile_shader(GL_VERTEX_SHADER, vertex_source);
    const GLuint fragment = compile_shader(GL_FRAGMENT_SHADER, fragment_source);
    GLuint program = 0;
    if (vertex && fragment) {
        program = glCreateProgram();
        glAttachShader(program, vertex);
        glAttachShader(program, fragment);
        glLinkProgram(program);
        GLint linked = 0;
        glGetProgramiv(program, GL_LINK_STATUS, &linked);
        if (!linked) {
            char message[1024]{};
            glGetProgramInfoLog(program, sizeof(message), nullptr, message);
            std::fprintf(stderr, "Terrain program: %s\n", message);
            glDeleteProgram(program);
            program = 0;
        }
    }
    if (vertex) glDeleteShader(vertex);
    if (fragment) glDeleteShader(fragment);
    return program;
}

GLuint load_texture(const std::filesystem::path& executable, const char* filename, int channels = 3,
                    TerrainGpu::ColorMap* samples = nullptr) {
    for (const auto& path : asset_paths(executable, filename)) {
        std::ifstream file(path, std::ios::binary);
        if (!file) continue;
        const std::vector<unsigned char> encoded((std::istreambuf_iterator<char>(file)), {});
        int width = 0;
        int height = 0;
        std::unique_ptr<unsigned char, decltype(&stbi_image_free)> pixels(
            stbi_load_from_memory(encoded.data(), static_cast<int>(encoded.size()), &width, &height, nullptr, channels),
            &stbi_image_free);
        if (!pixels) continue;

        GLint texture_limit = 0;
        glGetIntegerv(GL_MAX_TEXTURE_SIZE, &texture_limit);
        if (texture_limit <= 0) return 0;
        while (width > texture_limit || height > texture_limit) {
            const int half_width = width / 2;
            const int half_height = height / 2;
            for (int y = 0; y < half_height; ++y) {
                for (int x = 0; x < half_width; ++x) {
                    const int first = channels * (y * 2 * width + x * 2);
                    const int last = first + channels * width;
                    for (int channel = 0; channel < channels; ++channel) {
                        const auto* values = pixels.get() + channel;
                        pixels.get()[channels * (y * half_width + x) + channel] = static_cast<unsigned char>(
                            (values[first] + values[first + channels] + values[last] + values[last + channels]) / 4);
                    }
                }
            }
            width = half_width;
            height = half_height;
        }

        GLuint texture = 0;
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        constexpr GLenum rgb = 0x1907;
        constexpr GLenum red = 0x1903;
        constexpr GLenum red8 = 0x8229;
        glTexImage2D(GL_TEXTURE_2D, 0, channels == 1 ? red8 : rgb, width, height, 0,
                     channels == 1 ? red : rgb, GL_UNSIGNED_BYTE, pixels.get());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        using MipmapFunction = void (APIENTRY*)(GLenum);
        const auto generate_mipmaps = reinterpret_cast<MipmapFunction>(glfwGetProcAddress("glGenerateMipmap"));
        constexpr GLint trilinear = 0x2703;
        if (generate_mipmaps) generate_mipmaps(GL_TEXTURE_2D);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, generate_mipmaps ? trilinear : GL_LINEAR);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
        glBindTexture(GL_TEXTURE_2D, 0);
        if (glGetError() == 0) {
            if (samples) {
                const int step = std::max(1, width / 4096);
                samples->width = width / step;
                samples->height = height / step;
                samples->pixels.resize(static_cast<std::size_t>(samples->width) * samples->height * 3);
                for (int y = 0; y < samples->height; ++y) {
                    for (int x = 0; x < samples->width; ++x) {
                        const auto* from = pixels.get() + (y * step * width + x * step) * channels;
                        auto* to = samples->pixels.data() + (y * samples->width + x) * 3;
                        std::copy_n(from, 3, to);
                    }
                }
            }
            return texture;
        }
        glDeleteTextures(1, &texture);
    }
    return 0;
}

GLuint load_elevation(const std::filesystem::path& executable) {
    GLint texture_limit = 0;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &texture_limit);
    if (texture_limit <= 0) return 0;
    for (const auto& path : asset_paths(executable, "earth-elevation.png")) {
        std::ifstream file(path, std::ios::binary);
        if (!file) continue;
        const std::vector<unsigned char> encoded((std::istreambuf_iterator<char>(file)), {});
        int width = 0, height = 0;
        decltype(terrain.meters) pixels(stbi_load_16_from_memory(encoded.data(), static_cast<int>(encoded.size()),
                                                                &width, &height, nullptr, 1), stbi_image_free);
        if (!pixels) continue;
        while (width > texture_limit || height > texture_limit) {
            for (int y = 0; y < height / 2; ++y) {
                for (int x = 0; x < width / 2; ++x) {
                    const auto* sample = pixels.get() + 2 * y * width + 2 * x;
                    pixels.get()[y * (width / 2) + x] = (sample[0] + sample[1] + sample[width] + sample[width + 1]) / 4;
                }
            }
            width /= 2;
            height /= 2;
        }
        GLuint texture = 0;
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        constexpr GLenum red = 0x1903, red16 = 0x822A;
        glTexImage2D(GL_TEXTURE_2D, 0, red16, width, height, 0, red, GL_UNSIGNED_SHORT, pixels.get());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
        glBindTexture(GL_TEXTURE_2D, 0);
        if (glGetError() == 0) {
            terrain.elevation_width = width;
            terrain.elevation_height = height;
            terrain.meters = std::move(pixels);
            return texture;
        }
        glDeleteTextures(1, &texture);
    }
    return 0;
}

template<class Sample>
float bilinear(int width, int height, double u, double v, Sample at) {
    const double x = u * width - 0.5, y = std::clamp(v * height - 0.5, 0.0, height - 1.0);
    const int col = static_cast<int>(std::floor(x)), row = static_cast<int>(y);
    const int west = (col + width) % width, east = (west + 1) % width, south = std::min(row + 1, height - 1);
    return static_cast<float>(std::lerp(std::lerp(at(west, row), at(east, row), x - col),
                                       std::lerp(at(west, south), at(east, south), x - col), y - row));
}

float coarse_height(double u, double v) {
    const int width = terrain.elevation_width;
    return bilinear(width, terrain.elevation_height, u, v, [&](int x, int y) {
        return static_cast<double>(terrain.meters.get()[y * width + x]);
    });
}

Vec3 map_color(const TerrainGpu::ColorMap& map, double u, double v) {
    const auto channel = [&](int c) {
        return bilinear(map.width, map.height, u, v, [&](int x, int y) {
            return map.pixels[(y * map.width + x) * 3 + c] / 255.0;
        });
    };
    return {channel(0), channel(1), channel(2)};
}

struct TerrainDraw {
    std::array<float, 16> camera;
    std::array<float, 16> layout;
    ImVec2 clip_min;
    ImVec2 clip_max;
    bool local;
};

void render_terrain(const ImDrawList*, const ImDrawCmd* command) {
    const auto& data = *static_cast<const TerrainDraw*>(command->UserCallbackData);
    GLint viewport[4]{};
    glGetIntegerv(GL_VIEWPORT, viewport);
    const float scale_x = viewport[2] / data.layout[6], scale_y = viewport[3] / data.layout[7];
    const int left = std::clamp(static_cast<int>(std::floor((data.clip_min.x - data.layout[4]) * scale_x)), 0, viewport[2]);
    const int right = std::clamp(static_cast<int>(std::ceil((data.clip_max.x - data.layout[4]) * scale_x)), 0, viewport[2]);
    const int top = std::clamp(static_cast<int>(std::floor((data.clip_min.y - data.layout[5]) * scale_y)), 0, viewport[3]);
    const int bottom = std::clamp(static_cast<int>(std::ceil((data.clip_max.y - data.layout[5]) * scale_y)), 0, viewport[3]);
    if (left >= right || top >= bottom) return;
    glEnable(GL_SCISSOR_TEST);
    glScissor(viewport[0] + left, viewport[1] + viewport[3] - bottom, right - left, bottom - top);
    glUseProgram(terrain.program);
    glUniformMatrix4fv(terrain.camera, 1, GL_FALSE, data.camera.data());
    glUniformMatrix4fv(terrain.layout, 1, GL_FALSE, data.layout.data());
    glActiveTexture(GL_TEXTURE0 + 2);
    glBindTexture(GL_TEXTURE_2D, terrain.cities);
    glActiveTexture(GL_TEXTURE0 + 1);
    glBindTexture(GL_TEXTURE_2D, terrain.bathymetry);
    terrain.tiles->bind(terrain.texture);
    glActiveTexture(GL_TEXTURE0 + 6);
    glBindTexture(GL_TEXTURE_2D, terrain.elevation);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, terrain.texture);
    glBindVertexArray(terrain.vao);
    if (!data.local) {
        glUniform1i(terrain.mesh_mode, 0);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
    }
    glEnable(GL_DEPTH_TEST);
    glClear(depth_buffer_bit);
    glUniform1i(terrain.mesh_mode, data.local ? 2 : 1);
    const auto offset = static_cast<std::uintptr_t>(6 + (data.local ? globe_index_count : 0)) * sizeof(GLuint);
    glDrawElements(GL_TRIANGLES, data.local ? detail_index_count : globe_index_count, GL_UNSIGNED_INT,
                   reinterpret_cast<const void*>(offset));
    glDisable(GL_DEPTH_TEST);
}

} // namespace

bool initialize_earth_texture(const std::filesystem::path& executable, bool stream_tiles) {
    terrain.program = terrain_program();
    if (terrain.program) {
        terrain.texture = load_texture(executable, "earth-topography.jpg", 3, &terrain.satellite_samples);
        terrain.bathymetry = load_texture(executable, "ocean-relief.png", 3, &terrain.ocean_samples);
        terrain.cities = load_texture(executable, "city-lights.jpg", 1);
        terrain.elevation = load_elevation(executable);
    }
    if (!terrain.texture || !terrain.bathymetry || !terrain.cities || !terrain.elevation) {
        shutdown_earth_texture();
        return false;
    }
    terrain.camera = glGetUniformLocation(terrain.program, "camera");
    terrain.layout = glGetUniformLocation(terrain.program, "layoutData");
    terrain.mesh_mode = glGetUniformLocation(terrain.program, "meshMode");
    glUseProgram(terrain.program);
    glUniform1i(glGetUniformLocation(terrain.program, "atlas"), 0);
    glUniform1i(glGetUniformLocation(terrain.program, "oceanRelief"), 1);
    glUniform1i(glGetUniformLocation(terrain.program, "cityLights"), 2);
    glUniform1i(glGetUniformLocation(terrain.program, "satelliteTiles"), 3);
    glUniform1i(glGetUniformLocation(terrain.program, "elevationTiles"), 4);
    glUniform1i(glGetUniformLocation(terrain.program, "tilePages"), 5);
    glUniform1i(glGetUniformLocation(terrain.program, "globeElevation"), 6);
    glUseProgram(0);

    std::vector<GLuint> indices{0, 1, 2, 0, 2, 3};
    indices.reserve(6 + globe_index_count + detail_index_count);
    const auto add_grid = [&](GLuint columns, GLuint rows) {
        const auto stride = columns + 1;
        for (GLuint row = 0; row < rows; ++row) {
            for (GLuint col = 0; col < columns; ++col) {
                const auto a = row * stride + col;
                const auto b = a + stride;
                indices.insert(indices.end(), {a, a + 1, b, b, a + 1, b + 1});
            }
        }
    };
    add_grid(globe_columns, globe_rows);
    add_grid(detail_columns, detail_rows);
    glGenVertexArrays(1, &terrain.vao);
    glGenBuffers(1, &terrain.indices);
    glBindVertexArray(terrain.vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, terrain.indices);
    constexpr GLenum static_draw = 0x88E4;
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(indices[0]), indices.data(), static_draw);
    glBindVertexArray(0);
    if (glGetError() == 0) {
        terrain.tiles = std::make_unique<EarthTiles>(executable, stream_tiles);
        return true;
    }
    shutdown_earth_texture();
    return false;
}

void shutdown_earth_texture() {
    terrain.tiles.reset();
    if (terrain.vao) glDeleteVertexArrays(1, &terrain.vao);
    if (terrain.indices) glDeleteBuffers(1, &terrain.indices);
    if (terrain.texture) glDeleteTextures(1, &terrain.texture);
    if (terrain.bathymetry) glDeleteTextures(1, &terrain.bathymetry);
    if (terrain.cities) glDeleteTextures(1, &terrain.cities);
    if (terrain.elevation) glDeleteTextures(1, &terrain.elevation);
    if (terrain.program) glDeleteProgram(terrain.program);
    terrain = {};
}

bool earth_terrain_available() {
    return terrain.texture != 0;
}

bool earth_detail_visible() {
    return terrain.detail.side > 0.0f;
}

float earth_surface_height(Vec3 point, bool detail) {
    if (!terrain.meters) return 0.0f;
    const double horizontal = std::hypot(static_cast<double>(point.x), static_cast<double>(point.z));
    const double latitude = std::atan2(point.y, horizontal);
    const double u = horizontal > 0.00001 ? 0.5 + std::atan2(point.z, point.x) / (2.0 * std::numbers::pi) : 0.5;
    const double v = 0.5 - latitude / std::numbers::pi;
    const float meters = coarse_height(u, v);
    return detail && earth_detail_visible() ? terrain.tiles->elevation(terrain.detail, u, std::sin(latitude), meters) : meters;
}

std::uint64_t prepare_local_terrain(Vec3 anchor, float half_extent_m) {
    if (!earth_terrain_available()) return 0;
    const double longitude = std::atan2(anchor.z, anchor.x);
    const Vec3 east{-static_cast<float>(std::sin(longitude)), 0.0f, static_cast<float>(std::cos(longitude))};
    const Vec3 north{-anchor.y * east.z, std::hypot(anchor.x, anchor.z), anchor.y * east.x};
    terrain.detail = terrain.tiles->update({0.0f, 0.0f}, static_cast<float>(earth_radius_meters),
        {-half_extent_m, -half_extent_m}, {half_extent_m, half_extent_m}, east, north, anchor);
    return terrain.tiles->revision();
}

LocalSurface sample_local_terrain(double latitude, double longitude, bool detail) {
    if (!terrain.meters) return {0.0f, 0.0f, {.27f, .30f, .20f}};
    constexpr double pi = std::numbers::pi;
    const double u = 0.5 + std::remainder(longitude, 2.0 * pi) / (2.0 * pi);
    const double v = 0.5 - latitude / pi;
    const float base = coarse_height(u, v);
    const auto ocean = map_color(terrain.ocean_samples, u, v);
    auto color = map_color(terrain.satellite_samples, u, v);
    const float seabed = ocean.x * 8500.0f;
    const bool open_water = (ocean.z > .58f && seabed > 2.5f) ||
        (ocean.z > .37f && base < 2.0f && std::max(color.x, color.y) < .075f);
    float height = open_water ? -std::max(2.5f, seabed) : base;
    if (detail && earth_detail_visible()) {
        const double sin_lat = std::sin(latitude);
        height = terrain.tiles->elevation(terrain.detail, u, sin_lat, height, false);
        color = terrain.tiles->color(terrain.detail, u, sin_lat, color);
    }
    if (height < -1.5f && (open_water || ocean.z > .36f || color.z > color.x * 1.1f)) {
        return {0.0f, -height, color};
    }
    // Dark blue inland imagery can identify larger lakes above sea level.
    const bool lake = color.x < .17f && color.z > .07f && color.z > color.x * 1.38f && color.y > color.x * 1.13f;
    return {height, lake ? 4.0f : 0.0f, color};
}

bool draw_earth_texture(ImDrawList& draw, ImVec2 center, float radius, Vec3 east, Vec3 north, Vec3 forward,
                        Vec3 light, float pivot_y, float pivot_radius) {
    if (!terrain.texture) return false;
    const ImVec2 min = draw.GetClipRectMin();
    const ImVec2 max = draw.GetClipRectMax();
    const auto* viewport = ImGui::GetMainViewport();
    if (max.x <= min.x || max.y <= min.y) return true;
    const ImVec2 sphere_center{center.x, center.y + radius * pivot_y};
    const float margin_x = (max.x - min.x) * (pivot_y == 0.0f ? 0.0f : 0.14f);
    const float margin_y = (max.y - min.y) * (pivot_y == 0.0f ? 0.0f : 0.14f);
    terrain.detail = terrain.tiles->update(sphere_center, radius * pivot_radius, {min.x - margin_x, min.y - margin_y},
                                           {max.x + margin_x, max.y + margin_y}, east, north, forward);
    const auto detail = terrain.detail;

    // Pad the regional grid for the horizontal shift caused by radial displacement.
    const float max_radius = 1.0f + static_cast<float>(max_land_elevation / earth_radius_meters);
    const float x0 = (min.x - center.x) / radius, x1 = (max.x - center.x) / radius;
    const float y0 = (center.y - max.y) / radius + pivot_y, y1 = (center.y - min.y) / radius + pivot_y;
    const float left = std::min(x0, x0 / max_radius) - 1.0f / radius;
    const float right = std::max(x1, x1 / max_radius) + 1.0f / radius;
    const float bottom = std::min(y0, y0 / max_radius) - 1.0f / radius;
    const float top = std::max(y1, y1 / max_radius) + 1.0f / radius;
    const bool local = std::hypot(std::max(std::abs(left), std::abs(right)),
                                  std::max(std::abs(bottom), std::abs(top))) < 0.96f;
    TerrainDraw data{
        {east.x, east.y, east.z, 0, north.x, north.y, north.z, 0, forward.x, forward.y, forward.z, 0,
         center.x, center.y, radius, 0},
        {left, bottom, right, top, viewport->Pos.x, viewport->Pos.y, viewport->Size.x, viewport->Size.y,
         light.x, light.y, light.z, pivot_y, detail.x, detail.y, detail.side, 0},
        min, max, local
    };
    draw.AddCallback(render_terrain, &data, sizeof(data));
    draw.AddCallback(ImDrawCallback_ResetRenderState, nullptr);
    return true;
}

} // namespace wildlife
