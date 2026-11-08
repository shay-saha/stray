#include "colony_view.hpp"

#include "animal_geometry.hpp"
#include "earth_texture.hpp"

#include <imgui_impl_opengl3_loader.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <numbers>
#include <utility>

namespace wildlife {
namespace {

using namespace local;
using enum Locomotion;
constexpr float pi = std::numbers::pi_v<float>;
constexpr float tau = 2.0f * pi;
constexpr Vec3 up{0, 1, 0};
constexpr GLenum depth_buffer_bit = 0x0100;
constexpr GLenum static_draw = 0x88E4, stream_draw = 0x88E0;

Vec3 mix(Vec3 a, Vec3 b, float t) { return a + (b - a) * t; }

float hash(float value) {
    const float raw = std::sin(value * 91.237f + 51.84f) * 15731.789f;
    return raw - std::floor(raw);
}

struct Location {
    Vec3 center;
    Vec3 east;
    Vec3 north;

    explicit Location(Vec3 point) : center(unit(point)) {
        const float lon = std::atan2(point.z, point.x);
        east = {-std::sin(lon), 0.0f, std::cos(lon)};
        north = cross(east, center);
    }

    std::array<double, 3> at(float x, float z) const {
        const double e = x / earth_radius_meters, n = z / earth_radius_meters;
        return {center.x + e * east.x + n * north.x, center.y + n * north.y,
                center.z + e * east.z + n * north.z};
    }

    Vec3 offset(float x, float z) const {
        const auto p = at(x, z);
        return unit({static_cast<float>(p[0]), static_cast<float>(p[1]), static_cast<float>(p[2])});
    }

    LocalSurface sample(float x, float z, bool detail = true) const {
        const auto p = at(x, z);
        return sample_local_terrain(std::atan2(p[1], std::hypot(p[0], p[2])), std::atan2(p[2], p[0]), detail);
    }
};

std::optional<Vec3> suitable_anchor(Vec3 point, AnimalProfile profile) {
    const Location where(point);
    const auto mode = locomotion(profile.form);
    for (const float distance : {0.0f, 120.0f, 480.0f, 1600.0f, 4800.0f, 12000.0f, 35000.0f, 95000.0f, 180000.0f}) {
        const int steps = distance == 0.0f ? 1 : 12;
        for (int angle = 0; angle < steps; ++angle) {
            const float x = distance * std::sin(tau * angle / steps);
            const float z = distance * std::cos(tau * angle / steps);
            const auto surface = where.sample(x, z, false);
            if (mode == Swim) {
                if (surface.water_depth > std::max(.7f, profile.length * .75f)) return where.offset(x, z);
                continue;
            }
            if (surface.water_depth > .15f) {
                if (mode == WaterBird) return where.offset(x, z);
                continue;
            }
            const float gradient_x = where.sample(x + 30.0f, z, false).height - where.sample(x - 30.0f, z, false).height;
            const float gradient_z = where.sample(x, z + 30.0f, false).height - where.sample(x, z - 30.0f, false).height;
            if (std::hypot(gradient_x, gradient_z) < 60.0f * maximum_grade(profile.form)) return where.offset(x, z);
        }
    }
    return std::nullopt;
}

const auto& terrain_axis() {
    static const auto axis = [] {
        std::vector<float> positive;
        for (float x = 0; x <= 8000.0f; x += x < 45.0f ? 1.5f : x < 160.0f ? 6.0f : x < 550.0f ? 22.0f :
                                         x < 1800.0f ? 75.0f : 170.0f) positive.push_back(x);
        std::vector<float> result;
        for (auto x = positive.rbegin(); x != positive.rend() - 1; ++x) result.push_back(-*x);
        result.insert(result.end(), positive.begin(), positive.end());
        return result;
    }();
    return axis;
}

struct Grid {
    Location location;
    std::vector<LocalSurface> current, start, target;
    Mesh ground, water;
    float blend = 1.0f;
    std::uint64_t version = 0;
    double next_refresh = 0.0;
    bool dirty = true;

    explicit Grid(Vec3 point) : location(point) {}

    void refresh(bool first) {
        const auto& axis = terrain_axis();
        const auto datum = location.sample(0, 0).height;
        target.clear();
        target.reserve(axis.size() * axis.size());
        for (const float z : axis) {
            for (const float x : axis) {
                auto value = location.sample(x, z);
                value.height -= datum;
                target.push_back(value);
            }
        }
        start = first ? target : current;
        current = start;
        blend = first ? 1.0f : 0.0f;
        if (first) build();
    }

    void update(float dt, double now) {
        const auto revision = prepare_local_terrain(location.center, 2900.0f);
        if (revision != version && now >= next_refresh) {
            version = revision;
            next_refresh = now + .44;
            refresh(false);
        }
        if (blend < 1.0f) {
            blend = std::min(1.0f, blend + dt / .36f);
            const float t = blend * blend * (3.0f - 2.0f * blend);
            for (std::size_t i = 0; i < current.size(); ++i) {
                current[i] = {std::lerp(start[i].height, target[i].height, t),
                              std::lerp(start[i].water_depth, target[i].water_depth, t), mix(start[i].color, target[i].color, t)};
            }
            build();
        }
    }

    LocalSurface sample(float x, float z) const {
        const auto& axis = terrain_axis();
        const auto cell = [&](float p) {
            return std::clamp(static_cast<int>(std::upper_bound(axis.begin(), axis.end(), p) - axis.begin()) - 1,
                              0, static_cast<int>(axis.size()) - 2);
        };
        const int col = cell(x), row = cell(z);
        const float u = std::clamp((x - axis[col]) / (axis[col + 1] - axis[col]), 0.0f, 1.0f);
        const float v = std::clamp((z - axis[row]) / (axis[row + 1] - axis[row]), 0.0f, 1.0f);
        const auto i = row * axis.size() + col;
        const auto& a = current[i];
        const auto& b = current[i + 1];
        const auto& c = current[i + axis.size()];
        const auto& d = current[i + axis.size() + 1];
        const auto interpolate = [&](float LocalSurface::*field) {
            return u + v < 1.0f ? a.*field + (b.*field - a.*field) * u + (c.*field - a.*field) * v
                                 : d.*field + (c.*field - d.*field) * (1.0f - u) + (b.*field - d.*field) * (1.0f - v);
        };
        return {interpolate(&LocalSurface::height), interpolate(&LocalSurface::water_depth), a.color};
    }

    float grade(float x, float z, float stride) const {
        const float east = sample(x + stride, z).height - sample(x - stride, z).height;
        const float north = sample(x, z + stride).height - sample(x, z - stride).height;
        return std::hypot(east, north) / (2.0f * stride);
    }

    void build() {
        ground.clear();
        water.clear();
        const auto& axis = terrain_axis();
        const auto size = static_cast<std::uint32_t>(axis.size());
        const auto bottom = [&](std::uint32_t index) {
            return current[index].height - std::min(current[index].water_depth, 260.0f);
        };
        for (std::uint32_t row = 0; row < size; ++row) {
            for (std::uint32_t col = 0; col < size; ++col) {
                const auto i = row * size + col;
                const auto left = col > 0 ? col - 1 : col, right = std::min(col + 1, size - 1);
                const auto south = row > 0 ? row - 1 : row, north = std::min(row + 1, size - 1);
                const auto dx = (bottom(row * size + right) - bottom(row * size + left)) / (axis[right] - axis[left]);
                const auto dz = (bottom(north * size + col) - bottom(south * size + col)) / (axis[north] - axis[south]);
                const auto& s = current[i];
                const bool wet = s.water_depth > .24f;
                const auto land = mix(rgb(0x444933), s.color, .73f) * 1.07f;
                const auto bed = mix(rgb(0x435253), rgb(0x15303c), std::clamp(s.water_depth / 130.0f, 0.0f, 1.0f));
                ground.vertices.push_back({{axis[col], bottom(i), axis[row]}, unit({-dx, 1, -dz}), wet ? bed : land});
                water.vertices.push_back({{axis[col], s.height, axis[row]}, up, rgb(0x396376), .49f});
                if (col + 1 == size || row + 1 == size) continue;
                const std::array<std::uint32_t, 6> face{i, i + size, i + 1, i + 1, i + size, i + size + 1};
                ground.indices.insert(ground.indices.end(), face.begin(), face.end());
                for (int t = 0; t < 6; t += 3) {
                    if (current[face[t]].water_depth > .24f && current[face[t + 1]].water_depth > .24f &&
                        current[face[t + 2]].water_depth > .24f) {
                        water.indices.insert(water.indices.end(), face.begin() + t, face.begin() + t + 3);
                    }
                }
            }
        }
        // The satellite map supplies the ground color. Small tufts restore a sense of scale at eye level.
        for (int i = 0; i < 3400; ++i) {
            const float seed = static_cast<float>(i);
            const float x = (hash(seed + 1927.0f) - .5f) * 62.0f, z = (hash(seed + 4398.0f) - .5f) * 62.0f;
            const auto soil = sample(x, z);
            if (soil.water_depth > .1f || soil.color.z > soil.color.y + .018f ||
                soil.color.y < soil.color.x * .84f || soil.color.y > .67f || grade(x, z, .32f) > .47f) continue;
            const auto plant = mix(rgb(0x5d6343), rgb(0x8c8053), hash(seed + 7213.0f));
            const float y = soil.height - .014f;
            for (int blade = 0; blade < 3; ++blade) {
                const float angle = tau * (hash(seed + 911.0f) + blade / 3.0f);
                const float dx = std::cos(angle), dz = std::sin(angle);
                const float height = .060f + hash(seed + blade * 17.0f) * .11f;
                const auto index = static_cast<std::uint32_t>(ground.vertices.size());
                ground.vertices.push_back({{x + dx * .04f, y, z + dz * .04f}, up, plant * .63f});
                ground.vertices.push_back({{x + dx * .04f - dz * .019f, y, z + dz * .04f + dx * .019f}, up, plant * .73f});
                ground.vertices.push_back({{x + dx * .103f, y + height, z + dz * .103f}, up, plant});
                ground.indices.insert(ground.indices.end(), {index, index + 1, index + 2});
            }
        }
        dirty = true;
    }
};

struct Agent {
    AnimalId id;
    float x, z, yaw;
    float scale;
    float clock;
    float phase;
    float speed = 0.0f;
    float feeding = 0.0f;
    float airborne = 0.0f;
    bool swimming = false;
    bool visible = true;
    Activity activity = Activity::Resting;
    Vec3 position{};
};

struct Camera {
    Vec3 eye;
    Vec3 forward;
    Vec3 right;
    Vec3 vertical;
    float focal;
    float aspect;

    std::array<float, 16> matrix(float near, float far) const {
        const float a = (far + near) / (far - near), b = -2.0f * far * near / (far - near);
        const auto x = right * (focal / aspect), y = vertical * focal;
        const auto z = forward * a;
        return {x.x, y.x, z.x, forward.x, x.y, y.y, z.y, forward.y, x.z, y.z, z.z, forward.z,
                -dot(eye, x), -dot(eye, y), b - dot(eye, z), -dot(eye, forward)};
    }

    bool project(Vec3 point, ImVec2 min, ImVec2 size, ImVec2& screen) const {
        const auto rel = point - eye;
        const float z = dot(rel, forward);
        if (z <= 0.0f) return false;
        screen = {min.x + size.x * (.5f + dot(rel, right) * focal / (2.0f * z * aspect)),
                  min.y + size.y * (.5f - dot(rel, vertical) * focal / (2.0f * z))};
        return screen.x >= min.x && screen.x < min.x + size.x && screen.y >= min.y && screen.y < min.y + size.y;
    }
};

constexpr const char* vertex_source = R"glsl(#version 150
uniform mat4 viewProjection;
in vec3 position;
in vec3 normal;
in vec4 color;
in vec4 markings;
out vec3 world;
out vec3 surfaceNormal;
out vec4 surfaceColor;
out vec4 coatPoint;
void main() {
    world = position;
    surfaceNormal = normal;
    surfaceColor = color;
    coatPoint = markings;
    gl_Position = viewProjection * vec4(position, 1.0);
}
)glsl";
constexpr const char* fragment_source = R"glsl(#version 150
uniform mat4 lighting;
uniform int material;
in vec3 world;
in vec3 surfaceNormal;
in vec4 surfaceColor;
in vec4 coatPoint;
out vec4 result;
float grain(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}
float noise(vec2 p) {
    vec2 cell = floor(p);
    vec2 f = smoothstep(vec2(0.0), vec2(1.0), fract(p));
    return mix(mix(grain(cell), grain(cell + vec2(1,0)), f.x),
               mix(grain(cell + vec2(0,1)), grain(cell + vec2(1,1)), f.x), f.y);
}
void main() {
    vec3 base = surfaceColor.rgb;
    vec3 n = normalize(surfaceNormal);
    float distance = length(lighting[0].xyz - world);
    float strength = lighting[2].w;
    if (material == 0) {
        float texture = 0.78 + 0.20 * noise(world.xz * 5.4) + 0.17 * noise(world.xz * 1.3);
        base *= mix(texture, 1.0, smoothstep(70.0, 400.0, distance));
    }
    if (material == 1 && coatPoint.w > 0.5) {
        float paint;
        if (coatPoint.w < 1.5) {
            float stripe = sin(coatPoint.z * 28.0 + coatPoint.y * 4.8 + 0.85 * sin(coatPoint.x * 9.0));
            float edge = max(fwidth(stripe), 0.06);
            paint = smoothstep(-edge, edge, stripe);
        } else {
            float scale = coatPoint.w > 2.5 ? 7.0 : 12.0;
            float spot = noise(coatPoint.xz * scale + vec2(coatPoint.y * 4.8, coatPoint.y * 3.7));
            paint = smoothstep(0.58, 0.71, spot);
        }
        base *= mix(1.0, coatPoint.w > 2.5 ? 0.48 : 0.18, paint);
    }
    if (material == 3) {
        float wave = sin(world.x * 0.83 + world.z * 0.48) * 0.07 + noise(world.xz * 0.31) * 0.10;
        float rim = 1.0 - abs(dot(n, normalize(lighting[0].xyz - world)));
        float opacity = surfaceColor.a * (0.46 + rim * 0.50);
        result = vec4(base * (0.77 + wave) * (0.44 + strength), opacity);
        return;
    }
    if (material == 2) {
        result = surfaceColor;
        return;
    }
    float lambert = max(0.0, dot(n, lighting[2].xyz));
    float sky = 0.44 + 0.19 * max(0.0, n.y);
    vec3 shade = base * (sky + lambert * 0.61) * strength;
    shade += base * vec3(0.036, 0.056, 0.082) * (1.0 - strength * 0.6);
    float haze = 1.0 - exp(-distance * lighting[0].w);
    result = vec4(mix(shade, lighting[1].rgb, haze), surfaceColor.a);
}
)glsl";

GLuint compile(GLenum kind, const char* text) {
    const auto shader = glCreateShader(kind);
    glShaderSource(shader, 1, &text, nullptr);
    glCompileShader(shader);
    GLint compiled = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (compiled) return shader;
    char log[1024]{};
    glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
    std::fprintf(stderr, "Colony shader: %s\n", log);
    glDeleteShader(shader);
    return 0;
}

struct Gpu {
    struct Buffer {
        GLuint array = 0, vertices = 0, indices = 0;
        GLsizei count = 0;
    };
    GLuint program = 0;
    GLint matrix = -1, lighting = -1, material = -1;
    std::array<Buffer, 4> buffers{};

    ~Gpu() {
        for (auto& b : buffers) {
            if (b.array) glDeleteVertexArrays(1, &b.array);
            if (b.vertices) glDeleteBuffers(1, &b.vertices);
            if (b.indices) glDeleteBuffers(1, &b.indices);
        }
        if (program) glDeleteProgram(program);
    }

    bool initialize() {
        if (program) return true;
        const auto vertex = compile(GL_VERTEX_SHADER, vertex_source), fragment = compile(GL_FRAGMENT_SHADER, fragment_source);
        if (vertex && fragment) {
            program = glCreateProgram();
            glAttachShader(program, vertex);
            glAttachShader(program, fragment);
            glLinkProgram(program);
            GLint linked = 0;
            glGetProgramiv(program, GL_LINK_STATUS, &linked);
            if (!linked) {
                char log[1024]{};
                glGetProgramInfoLog(program, sizeof(log), nullptr, log);
                std::fprintf(stderr, "Colony program: %s\n", log);
                glDeleteProgram(program);
                program = 0;
            }
        }
        if (vertex) glDeleteShader(vertex);
        if (fragment) glDeleteShader(fragment);
        if (!program) return false;
        matrix = glGetUniformLocation(program, "viewProjection");
        lighting = glGetUniformLocation(program, "lighting");
        material = glGetUniformLocation(program, "material");
        const auto bind = [&](const char* name, GLint size, std::size_t offset) {
            const auto attrib = static_cast<GLuint>(glGetAttribLocation(program, name));
            glEnableVertexAttribArray(attrib);
            glVertexAttribPointer(attrib, size, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<const void*>(offset));
        };
        for (auto& buffer : buffers) {
            glGenVertexArrays(1, &buffer.array);
            glGenBuffers(1, &buffer.vertices);
            glGenBuffers(1, &buffer.indices);
            glBindVertexArray(buffer.array);
            glBindBuffer(GL_ARRAY_BUFFER, buffer.vertices);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer.indices);
            bind("position", 3, offsetof(Vertex, position));
            bind("normal", 3, offsetof(Vertex, normal));
            bind("color", 4, offsetof(Vertex, color));
            bind("markings", 4, offsetof(Vertex, markings));
        }
        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        return true;
    }

    void upload(int slot, const Mesh& mesh, GLenum usage) {
        auto& b = buffers[slot];
        b.count = static_cast<GLsizei>(mesh.indices.size());
        if (!b.count) return;
        glBindVertexArray(b.array);
        glBindBuffer(GL_ARRAY_BUFFER, b.vertices);
        glBufferData(GL_ARRAY_BUFFER, mesh.vertices.size() * sizeof(Vertex), mesh.vertices.data(), usage);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.indices.size() * sizeof(std::uint32_t), mesh.indices.data(), usage);
    }

    void draw(int slot) const {
        const auto& b = buffers[slot];
        if (b.count == 0) return;
        glUniform1i(material, slot);
        glBindVertexArray(b.array);
        glDrawElements(GL_TRIANGLES, b.count, GL_UNSIGNED_INT, nullptr);
    }
};

const char* status(const Agent& a, AnimalProfile profile, AnimalGroup group) {
    if (a.swimming) return a.activity == Activity::Foraging ? "Foraging in water" : a.speed > .04f ? "Swimming" : "Resting in water";
    if (a.airborne > .25f) return "Flying";
    if (a.activity == Activity::Foraging) return group == AnimalGroup::Grazer ? "Grazing" : "Foraging";
    if (a.speed > .04f) return profile.form == AnimalForm::Wallaby ? "Hopping" : "Walking";
    return "Idling";
}

} // namespace

struct ColonyView::State {
    std::unique_ptr<Grid> terrain;
    Gpu gpu;
    std::vector<Agent> agents;
    Mesh creatures, shadows;
    AnimalProfile profile;
    std::size_t species = 0, followed = 0;
    bool playing = true;
    bool live = false;
    bool shifted = false;
    bool drag = false;
    float azimuth = 1.96f, elevation = .20f, distance = 9.0f;
    Vec3 target{};

    bool compatible(float x, float z, bool swimming, float spread = 0.0f) const {
        const auto sample = terrain->sample(x, z);
        if (swimming) return sample.water_depth > std::max(.46f, profile.length * .52f);
        if (locomotion(profile.form) == WaterBird && sample.water_depth > .28f) return true;
        if (profile.form == AnimalForm::Crocodile && sample.water_depth > profile.length * .30f) return true;
        if (sample.water_depth > .14f) return false;
        const float footprint = std::max(profile.length * .34f, spread);
        return terrain->grade(x, z, footprint) <= maximum_grade(profile.form) &&
            terrain->sample(x + footprint, z).water_depth < .15f && terrain->sample(x - footprint, z).water_depth < .15f &&
            terrain->sample(x, z + footprint).water_depth < .15f && terrain->sample(x, z - footprint).water_depth < .15f;
    }

    bool place(Agent& a, float x, float z, bool swimming) const {
        for (int ring = 0; ring < 12; ++ring) {
            const float r = ring * std::max(profile.length * 1.08f, 1.35f);
            for (int s = 0; s < (ring == 0 ? 1 : 12); ++s) {
                const float angle = tau * s / 12.0f + a.yaw;
                const float east = x + std::cos(angle) * r, north = z + std::sin(angle) * r;
                if (!compatible(east, north, swimming)) continue;
                a.x = east;
                a.z = north;
                return true;
            }
        }
        return false;
    }

    void animate(float dt) {
        const auto mode = locomotion(profile.form);
        const bool swim = mode == Swim;
        const bool flies = (mode == Flight || mode == WaterBird) && profile.form != AnimalForm::Penguin;
        const float range = std::max(26.0f, profile.length * 10.0f);
        for (std::size_t i = 0; i < agents.size(); ++i) {
            auto& a = agents[i];
            if (!compatible(a.x, a.z, swim) && a.airborne < .25f) {
                a.visible = place(a, a.x, a.z, swim);
            } else {
                a.visible = true;
            }
            if (!a.visible) continue;
            a.clock = std::fmod(a.clock + dt, 27.0f + hash(static_cast<float>(a.id.individual)) * 9.0f);
            a.activity = a.clock < 9.3f ? Activity::Traveling : a.clock < 20.3f ? Activity::Foraging : Activity::Resting;
            const auto floor = terrain->sample(a.x, a.z);
            const bool wet = floor.water_depth > .29f;
            const bool flight = flies && (profile.form == AnimalForm::Bat || a.activity == Activity::Traveling || (!compatible(a.x, a.z, false)));
            a.swimming = swim || (mode == WaterBird && wet && a.airborne < .2f) ||
                (profile.form == AnimalForm::Crocodile && floor.water_depth > profile.length * .3f);
            const float movement = a.activity == Activity::Traveling ? 1.0f : a.activity == Activity::Foraging ? .23f : 0.0f;
            const float pace = profile.form == AnimalForm::Tortoise ? .21f :
                swim ? std::clamp(profile.length * .53f, .21f, 3.8f) :
                flight ? std::clamp(profile.length * 3.8f, 1.35f, 5.5f) : std::clamp(profile.length * .69f, .13f, 1.75f);
            const float ease = 1.0f - std::exp(-5.0f * dt);
            const float desired = pace * (flight ? std::max(.67f, movement) : movement);
            a.speed = std::lerp(a.speed, desired, ease);
            a.feeding = std::lerp(a.feeding, a.activity == Activity::Foraging && !flight ? 1.0f : 0.0f, ease);
            a.airborne = std::lerp(a.airborne, flight ? 1.0f : 0.0f, 1.0f - std::exp(-1.25f * dt));
            if (a.speed > .001f && dt > 0.0f) {
                const float distance = std::hypot(a.x, a.z);
                const float aim = distance > range ? std::atan2(-a.x, -a.z) : a.yaw + std::sin(a.clock * .63f + i) * .14f;
                bool moved = false;
                for (const float bend : {0.0f, .54f, -.54f, 1.06f, -1.06f, 1.7f, -1.7f}) {
                    const float heading = aim + bend;
                    const float turn = std::remainder(heading - a.yaw, tau);
                    const float look = std::max(profile.length * .48f, a.speed * .7f);
                    const auto allowed = [&](float angle, float ahead) {
                        const float x = a.x + std::sin(angle) * ahead, z = a.z + std::cos(angle) * ahead;
                        if (std::hypot(x, z) > range * 1.32f) return false;
                        if (a.airborne > .66f && !swim) return true;
                        return compatible(x, z, swim);
                    };
                    if (!allowed(heading, look)) continue;
                    a.yaw += std::clamp(turn, -dt * 1.8f, dt * 1.8f);
                    const float advance = a.speed * dt * (std::abs(turn) > .82f ? .19f : 1.0f);
                    const float x = a.x + std::sin(a.yaw) * advance, z = a.z + std::cos(a.yaw) * advance;
                    const bool separated = std::none_of(agents.begin(), agents.end(), [&](const Agent& other) {
                        return &other != &a && other.visible && std::hypot(x - other.x, z - other.z) < profile.length * .68f;
                    });
                    if (allowed(a.yaw, advance) && separated) {
                        a.x = x;
                        a.z = z;
                        const float gait = profile.form == AnimalForm::Wallaby ? .91f : .68f;
                        a.phase += advance / (a.scale * gait);
                        moved = true;
                    }
                    break;
                }
                if (!moved) {
                    a.yaw += dt * (.7f + i * .09f);
                    a.speed *= std::exp(-6.0f * dt);
                }
            }
            const auto surface = terrain->sample(a.x, a.z);
            float height = surface.height;
            if (swim) {
                const float safe = a.scale * .24f;
                const float dive = a.scale * (.55f + .17f * std::sin(a.clock * .36f + i) + a.feeding * .16f);
                height -= std::clamp(dive, safe, std::max(safe, surface.water_depth - safe));
            } else if (a.airborne > .02f) {
                height += a.airborne * (std::max(1.5f, a.scale * 3.3f) + .26f * std::sin(a.clock * 2.2f + i));
            } else if (a.swimming) {
                height -= a.scale * .10f;
                height += a.scale * .026f * std::sin(a.clock * 2.4f + i);
            } else if (profile.form == AnimalForm::Wallaby) {
                height += a.scale * .15f * std::max(0.0f, std::sin(a.phase * tau)) * std::min(1.0f, a.speed);
            }
            a.position = {a.x, height, a.z};
        }
    }

    void build_animals() {
        creatures.clear();
        shadows.clear();
        const GroundHeight surface = [&](float x, float z) { return terrain->sample(x, z).height; };
        const auto& form = profile.form;
        for (const auto& a : agents) {
            if (!a.visible) continue;
            const float moving = std::clamp(a.speed / std::max(.15f, a.scale * .56f), 0.0f, 1.0f);
            const float phase = a.airborne > .35f ? a.clock * (form == AnimalForm::Bat ? 4.5f : 2.65f)
                              : a.swimming ? a.phase + a.clock * .52f : a.phase + a.clock * .013f;
            build_animal(creatures, *model_species, profile,
                {a.position, a.yaw, phase, moving, a.feeding, a.airborne, a.swimming, a.scale}, surface);
            if (a.swimming || a.airborne > .75f) continue;
            constexpr int points = 16;
            const auto base = static_cast<std::uint32_t>(shadows.vertices.size());
            const Vec3 color{.044f, .049f, .046f};
            const float center_x = a.x + a.scale * .20f, center_z = a.z - a.scale * .14f;
            shadows.vertices.push_back({{center_x, surface(center_x, center_z) + .014f, center_z}, up, color, .49f * (1.0f - a.airborne)});
            for (int j = 0; j < points; ++j) {
                const float angle = tau * j / points;
                const float x = center_x + a.scale * std::sin(angle) * .58f;
                const float z = center_z + a.scale * std::cos(angle) * .80f;
                shadows.vertices.push_back({{x, surface(x, z) + .014f, z}, up, color, 0.0f});
                for (const auto idx : {base, base + 1 + j, base + 1 + (j + 1) % points}) shadows.indices.push_back(idx);
            }
        }
    }

    const Species* model_species = nullptr;

    struct Render {
        State* scene;
        std::array<float,16> matrix, light;
        ImVec2 min, size, viewport_pos, viewport_size;
    };

    static void render(const ImDrawList*, const ImDrawCmd* command) {
        const auto& args = *static_cast<const Render*>(command->UserCallbackData);
        auto& scene = *args.scene;
        auto& gpu = scene.gpu;
        GLint viewport[4]{};
        glGetIntegerv(GL_VIEWPORT, viewport);
        const float sx = viewport[2] / args.viewport_size.x, sy = viewport[3] / args.viewport_size.y;
        const int left = static_cast<int>((args.min.x - args.viewport_pos.x) * sx);
        const int top = static_cast<int>((args.min.y - args.viewport_pos.y) * sy);
        const int width = static_cast<int>(args.size.x * sx), height = static_cast<int>(args.size.y * sy);
        if (width <= 0 || height <= 0) return;
        glEnable(GL_SCISSOR_TEST);
        glScissor(viewport[0] + left, viewport[1] + viewport[3] - top - height, width, height);
        glViewport(viewport[0] + left, viewport[1] + viewport[3] - top - height, width, height);
        glEnable(GL_DEPTH_TEST);
        glClear(depth_buffer_bit);
        glUseProgram(gpu.program);
        glUniformMatrix4fv(gpu.matrix, 1, GL_FALSE, args.matrix.data());
        glUniformMatrix4fv(gpu.lighting, 1, GL_FALSE, args.light.data());
        if (scene.terrain->dirty) {
            gpu.upload(0, scene.terrain->ground, static_draw);
            gpu.upload(3, scene.terrain->water, static_draw);
            scene.terrain->dirty = false;
        }
        gpu.upload(1, scene.creatures, stream_draw);
        gpu.upload(2, scene.shadows, stream_draw);
        for (const auto slot : {0, 2, 1, 3}) gpu.draw(slot);
        glDisable(GL_DEPTH_TEST);
        glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
    }
};

ColonyView::ColonyView() : state_(std::make_unique<State>()) {}
ColonyView::~ColonyView() = default;

bool ColonyView::open(const Simulation& simulation, AnimalId animal) {
    auto& s = *state_;
    if (!earth_terrain_available() || !s.gpu.initialize()) return false;
    const auto& colony = simulation.colonies()[animal.colony];
    const auto& species = simulation.species()[colony.species];
    const auto sample = simulation.sample_animal(animal, simulation.elapsed_hours());
    const auto profile = animal_profile(species);
    auto anchor = suitable_anchor(sample.position, profile);
    if (!anchor) anchor = suitable_anchor(simulation.colony_home(animal.colony), profile);
    if (!anchor && locomotion(profile.form) == Swim) {
        anchor = suitable_anchor(earth_position(38.9f, -47.3f), profile);
    }
    if (!anchor) return false;
    s.shifted = length(*anchor - sample.position) * earth_radius_meters > 180.0;
    s.terrain = std::make_unique<Grid>(*anchor);
    s.terrain->version = prepare_local_terrain(*anchor, 2900.0f);
    s.terrain->refresh(true);
    s.model_species = &species;
    s.profile = profile;
    s.species = colony.species;
    s.agents.clear();
    const auto available = std::min<std::size_t>(colony.population, Simulation::max_visible_animals);
    const auto capacity = species.behavior == Behavior::Solitary ? 3u : species.behavior == Behavior::Pack ? 6u : 9u;
    const auto count = std::min<std::size_t>(available, capacity);
    for (std::size_t i = 0; i < count; ++i) {
        const AnimalId id{animal.colony, (animal.individual + i) % available};
        const float noise = hash(static_cast<float>(animal.colony * 577 + id.individual));
        const float radius = i == 0 ? 0.0f : (std::sqrt(static_cast<float>(i)) + .45f) * profile.length * 1.62f;
        const float angle = 2.399963f * i + noise * .24f;
        const auto activity = simulation.sample_animal(id, simulation.elapsed_hours()).activity;
        const float clock = (activity == Activity::Traveling ? 1.0f : activity == Activity::Foraging ? 10.0f : 22.0f);
        Agent a{id, 0, 0, i == 0 ? .44f : angle + 1.1f, profile.length * std::lerp(.83f, 1.11f, noise),
                std::fmod(clock + i * 5.41f, 27.0f), noise};
        a.visible = s.place(a, radius * std::sin(angle), radius * std::cos(angle), locomotion(profile.form) == Swim);
        s.agents.push_back(a);
    }
    s.followed = 0;
    s.azimuth = colony.region == "Matterhorn meadows" ? 1.21f : 1.87f;
    s.agents.front().yaw = s.azimuth - pi * .5f;
    s.elevation = locomotion(profile.form) == Swim ? .02f : .15f;
    s.distance = profile.length * 6.65f;
    s.playing = true;
    s.drag = false;
    s.live = true;
    s.animate(0.0f);
    s.target = s.agents.front().position + up * (profile.length * .45f);
    return true;
}

void ColonyView::close() { state_->live = false; }
bool ColonyView::active() const { return state_->live; }
void ColonyView::toggle_playback() { state_->playing = !state_->playing; }

std::optional<AnimalId> ColonyView::draw(const Simulation& simulation, bool night) {
    auto& s = *state_;
    if (ImGui::SmallButton(s.playing ? "Pause animals" : "Play animals")) s.playing = !s.playing;
    ImGui::SameLine();
    char followed[48];
    std::snprintf(followed, sizeof(followed), "Follow %zu", s.agents[s.followed].id.individual + 1);
    ImGui::SetNextItemWidth(108.0f);
    int selection = static_cast<int>(s.followed);
    if (ImGui::BeginCombo("##follow_animal", followed)) {
        for (std::size_t i = 0; i < s.agents.size(); ++i) {
            char label[40];
            std::snprintf(label, sizeof(label), "Animal %zu", s.agents[i].id.individual + 1);
            if (ImGui::Selectable(label, i == s.followed)) selection = static_cast<int>(i);
        }
        ImGui::EndCombo();
    }
    s.followed = static_cast<std::size_t>(selection);
    ImGui::SameLine();
    ImGui::TextDisabled("Drag to orbit  |  Scroll to zoom");
    const auto& io = ImGui::GetIO();
    const float dt = std::clamp(io.DeltaTime, 0.0f, .05f);
    const float zoom_unit = s.profile.length;
    s.terrain->update(dt, ImGui::GetTime());
    s.animate(s.playing ? dt : 0.0f);
    s.build_animals();

    const auto size = ImGui::GetContentRegionAvail();
    if (size.x <= 2.0f || size.y <= 2.0f) return s.agents[s.followed].id;
    const auto min = ImGui::GetCursorScreenPos();
    const ImVec2 max{min.x + size.x, min.y + size.y};
    ImGui::InvisibleButton("##colony_canvas", size);
    if (ImGui::IsItemActivated()) s.drag = false;
    if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 2.5f)) {
        s.drag = true;
        s.azimuth -= io.MouseDelta.x * .009f;
        s.elevation = std::clamp(s.elevation + io.MouseDelta.y * .008f, -.28f, 1.25f);
    }
    if (ImGui::IsItemHovered() && io.MouseWheel != 0) {
        s.distance = std::clamp(s.distance * std::pow(.84f, io.MouseWheel), zoom_unit * 2.15f, zoom_unit * 34.0f + 16.0f);
    }
    const auto& actor = s.agents[s.followed];
    const float raise = actor.swimming ? 0.0f : s.profile.form == AnimalForm::Giraffe ? .84f : .49f;
    const auto focus = actor.position + up * (actor.scale * raise);
    s.target = mix(s.target, focus, 1.0f - std::exp(-9.0f * dt));
    const float horizontal = s.distance * std::cos(s.elevation);
    auto eye = s.target + Vec3{horizontal * std::sin(s.azimuth), s.distance * std::sin(s.elevation),
                               horizontal * std::cos(s.azimuth)};
    const auto floor = s.terrain->sample(eye.x, eye.z);
    eye.y = std::max(eye.y, floor.height - std::min(floor.water_depth, 260.0f) + std::max(.30f, zoom_unit * .08f));
    const auto look = unit(s.target - eye);
    const auto right = unit(cross(look, up));
    const Camera camera{eye, look, right, cross(right, look), 2.25f, size.x / size.y};
    const auto& species = simulation.species()[s.species];
    const bool underwater = floor.water_depth > .42f && eye.y < floor.height;
    const Vec3 sky = underwater ? rgb(night ? 0x102734 : 0x315965) : rgb(night ? 0x17232d : 0x71858a);
    const auto direction = unit(Vec3{-.54f, .79f, .38f});
    const auto* viewport = ImGui::GetMainViewport();
    State::Render args{&s, camera.matrix(std::max(.015f, zoom_unit * .013f), 15000.0f),
                       {eye.x, eye.y, eye.z, underwater ? .26f / std::max(2.4f, zoom_unit * 4.4f) : .00051f,
                        sky.x, sky.y, sky.z, 0.0f, direction.x, direction.y, direction.z, night ? .50f : 1.02f, 0, 0, 0, 0},
                        min, size, viewport->Pos, viewport->Size};
    auto& draw = *ImGui::GetWindowDrawList();
    const auto sky_color = [&](float intensity) {
        return IM_COL32(static_cast<int>(255.0f * sky.x * intensity), static_cast<int>(255.0f * sky.y * intensity),
                        static_cast<int>(255.0f * sky.z * intensity), 255);
    };
    draw.AddRectFilledMultiColor(min, max, sky_color(.72f), sky_color(.72f), sky_color(1.0f), sky_color(1.0f));
    draw.AddCallback(State::render, &args, sizeof(args));
    draw.AddCallback(ImDrawCallback_ResetRenderState, nullptr);
    draw.PushClipRect(min, max, true);
    char name[144], info[160];
    std::snprintf(name, sizeof(name), "%s  |  Animal %zu", species.name.data(), actor.id.individual + 1);
    std::snprintf(info, sizeof(info), "%s   ·   %.1f m camera%s", actor.visible ? status(actor, s.profile, species.group) : "Seeking suitable terrain",
                  s.distance, s.shifted ? "   ·   Nearby habitat" : "");
    const auto a = ImGui::CalcTextSize(name), b = ImGui::CalcTextSize(info);
    const ImVec2 top{min.x + 12.0f, min.y + 12.0f};
    draw.AddRectFilled(top, {top.x + std::max(a.x, b.x) + 18.0f, top.y + a.y + b.y + 14.0f}, IM_COL32(8, 15, 20, 215), 4.0f);
    draw.AddText({top.x + 9.0f, top.y + 6.0f}, IM_COL32(234, 237, 233, 255), name);
    draw.AddText({top.x + 9.0f, top.y + 7.0f + a.y}, IM_COL32(171, 188, 190, 255), info);
    if (earth_detail_visible()) {
        draw.AddText({min.x + 14.0f, max.y - 24.0f}, IM_COL32(235, 236, 224, 225), "EOxCloudless by EOX · Mapzen elevation");
    }
    if (!s.drag && ImGui::IsItemDeactivated() && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
        float best = 52.0f;
        for (std::size_t i = 0; i < s.agents.size(); ++i) {
            ImVec2 screen;
            if (!s.agents[i].visible || !camera.project(s.agents[i].position + up * (s.agents[i].scale * raise), min, size, screen)) continue;
            const float d = std::hypot(io.MousePos.x - screen.x, io.MousePos.y - screen.y);
            if (d < best) {
                best = d;
                s.followed = i;
            }
        }
    }
    draw.PopClipRect();
    return s.agents[s.followed].id;
}

} // namespace wildlife
