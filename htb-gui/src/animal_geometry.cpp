#include "animal_geometry.hpp"

#include <algorithm>
#include <array>
#include <numbers>

namespace wildlife::local {
namespace {

constexpr float pi = std::numbers::pi_v<float>;
constexpr float tau = 2.0f * pi;
constexpr Vec3 up{0.0f, 1.0f, 0.0f};
constexpr Vec3 forward{0.0f, 0.0f, 1.0f};
constexpr Vec3 dark{.13f, .105f, .09f};
constexpr Vec3 ivory{.79f, .75f, .65f};
using enum AnimalForm;

float smooth(float x) { return x * x * (3.0f - 2.0f * x); }
float fractional(float x) { return x - std::floor(x); }
Vec3 mix(Vec3 a, Vec3 b, float t) { return a + (b - a) * t; }

struct Sphere {
    int rings, sides;
    std::vector<Vec3> points;
    std::vector<std::uint32_t> indices;
    Sphere(int r, int s) : rings(r), sides(s), points((r+1)*(s+1)), indices(r*s*6) {
        std::size_t index = 0;
        for (int r = 0; r <= rings; ++r) {
            const float polar = r * pi / rings;
            for (int s = 0; s <= sides; ++s) {
                const float theta = s * tau / sides;
                const auto a = static_cast<std::uint32_t>(r * (sides + 1) + s);
                points[a] = {std::sin(polar) * std::cos(theta), std::cos(polar), std::sin(polar) * std::sin(theta)};
                if (r < rings && s < sides) {
                    for (const auto value : {a, a + 1, a + sides + 1, a + 1, a + sides + 2, a + sides + 1}) {
                        indices[index++] = value;
                    }
                }
            }
        }
    }
};

struct Rig {
    Mesh& mesh;
    const AnimalPose& pose;
    Vec3 east;
    Vec3 ahead;
    Vec3 coat;
    Vec3 accent;
    Vec3 point(Vec3 v) const { return pose.origin + direction(v) * pose.scale; }
    Vec3 direction(Vec3 v) const { return east * v.x + up * v.y + ahead * v.z; }

    void ball(Vec3 center, Vec3 radii, Vec3 color, Vec3 axis = forward, float pattern = 0.0f) {
        static const std::array spheres{Sphere{3, 6}, Sphere{6, 10}, Sphere{12, 20}, Sphere{20, 32}, Sphere{32, 48}};
        const auto& sphere = spheres[std::min(pose.detail, 4u)];
        const Vec3 z = unit(axis);
        const Vec3 x = unit(cross(std::abs(z.y) > .99f ? Vec3{0, 0, 1} : up, z));
        const Vec3 y = cross(z, x);
        const auto base = static_cast<std::uint32_t>(mesh.vertices.size());
        for (const auto n : sphere.points) {
            const Vec3 p = center + x * (n.x * radii.x) + y * (n.y * radii.y) + z * (n.z * radii.z);
            const Vec3 normal = x * (n.x / radii.x) + y * (n.y / radii.y) + z * (n.z / radii.z);
            mesh.vertices.push_back({point(p), unit(direction(normal)), color * (1.0f + .07f * n.y),
                                     1.0f, {n.x, n.y, n.z, pattern}});
        }
        for (const auto index : sphere.indices) mesh.indices.push_back(base + index);
    }

    void tube(Vec3 a, Vec3 b, float start, float end, Vec3 color, float pattern = 0.0f) {
        constexpr int sides = 10;
        static const auto circle = [] {
            std::array<Vec2, sides> vertices;
            for (int i = 0; i < sides; ++i) vertices[i] = {std::cos(i * tau / sides), std::sin(i * tau / sides)};
            return vertices;
        }();
        const auto along = unit(b - a);
        const auto right = unit(cross(std::abs(along.y) > .93f ? forward : up, along));
        const auto other = cross(along, right);
        const auto base = static_cast<std::uint32_t>(mesh.vertices.size());
        for (int i = 0; i < sides; ++i) {
            const auto radial = right * circle[i].x + other * circle[i].y;
            const auto normal = unit(direction(radial + along * ((start - end) / std::max(.001f, length(b - a)))));
            mesh.vertices.push_back({point(a + radial * start), normal, color, 1.0f, {circle[i].x, circle[i].y, 0.0f, pattern}});
            mesh.vertices.push_back({point(b + radial * end), normal, color, 1.0f, {circle[i].x, circle[i].y, .85f, pattern}});
            const auto here = base + 2 * i, next = base + 2 * ((i + 1) % sides);
            for (const auto index : {here, next, here + 1, next, next + 1, here + 1}) mesh.indices.push_back(index);
        }
    }

    void blade(Vec3 root, Vec3 tip, Vec3 trailing, float thickness, Vec3 color) {
        const auto n = unit(cross(tip - root, trailing - root));
        const auto midpoint = root + (tip + trailing - root * 2.0f) * .30f;
        const auto top = midpoint + n * thickness, bottom = midpoint - n * thickness;
        const std::array<Vec3, 3> outline{root, tip, trailing};
        for (std::size_t i = 0; i < outline.size(); ++i) {
            const auto a = outline[i], b = outline[(i + 1) % outline.size()];
            for (const bool upper : {true, false}) {
                const auto c = upper ? top : bottom;
                const auto normal = unit(direction(upper ? cross(b - a, c - a) : cross(a - b, c - b)));
                const auto base = static_cast<std::uint32_t>(mesh.vertices.size());
                for (const auto v : {a, b, c}) mesh.vertices.push_back({point(v), normal, color});
                for (unsigned j = 0; j < 3; ++j) mesh.indices.push_back(base + j);
            }
        }
    }

    void horn(Vec3 root, float side, float height, float length_m, float thickness) {
        auto previous = root;
        for (int i = 1; i <= 11; ++i) {
            const auto curve = [&](float t) { return std::max(.022f, thickness * (1.0f - t)); };
            const float t = i / 11.0f, before = (i - 1) / 11.0f;
            const auto next = root + Vec3{side * (.12f * t), height * std::sin(t * 2.30f), -length_m * t * t};
            tube(previous, next, curve(before), i == 11 ? .001f : curve(t), accent * (i % 2 == 0 ? .81f : .96f));
            previous = next;
        }
    }
};

struct Shape {
    float hip, belly, width, trunk, head_y, head_z, muzzle, shin, tail;
};

Shape quadruped_shape(AnimalForm form) {
    switch (form) {
        case Ibex: return {.47f,.19f,.177f,.39f,.81f,.44f,.15f,.037f,.12f};
        case Deer: return {.52f,.185f,.158f,.37f,.84f,.46f,.16f,.035f,.17f};
        case Horse: case Zebra: return {.49f,.22f,.17f,.39f,.86f,.48f,.18f,.046f,.43f};
        case Bison: return {.34f,.26f,.236f,.40f,.63f,.49f,.19f,.055f,.24f};
        case Elephant: return {.41f,.287f,.225f,.42f,.68f,.48f,.18f,.083f,.29f};
        case Rhino: return {.29f,.26f,.207f,.42f,.47f,.49f,.22f,.064f,.16f};
        case Giraffe: return {.72f,.25f,.198f,.39f,1.41f,.49f,.16f,.048f,.29f};
        case Camel: return {.51f,.22f,.178f,.41f,1.0f,.50f,.16f,.046f,.26f};
        case Boar: return {.27f,.227f,.207f,.42f,.38f,.49f,.21f,.047f,.15f};
        case Tapir: return {.30f,.233f,.21f,.40f,.48f,.46f,.21f,.050f,.084f};
        case Felid: case Lion: return {.36f,.17f,.153f,.41f,.55f,.44f,.15f,.043f,.60f};
        case Canid: return {.38f,.178f,.139f,.37f,.64f,.44f,.20f,.039f,.44f};
        case Bear: return {.30f,.251f,.21f,.42f,.59f,.49f,.19f,.066f,.074f};
        case Hyena: return {.43f,.20f,.153f,.37f,.58f,.44f,.18f,.045f,.27f};
        case Primate: return {.31f,.20f,.145f,.35f,.60f,.43f,.13f,.048f,.57f};
        case SmallMammal: return {.18f,.17f,.155f,.40f,.34f,.48f,.15f,.039f,.34f};
        case Crocodile: case Lizard: return {.105f,.11f,.147f,.35f,.20f,.52f,.28f,.038f,.72f};
        default: return {.47f,.19f,.164f,.38f,.76f,.43f,.15f,.039f,.19f};
    }
}

void foot(Rig& rig, Vec3 hip, float x, float z, float phase, float leg_length, float radius,
          bool rear, const GroundHeight& height, float pattern = 0.0f) {
    const float moving = rig.pose.moving;
    const float stride = .21f * std::min(leg_length / .42f, 1.35f) * moving;
    const float swing = std::clamp((phase - .64f) / .36f, 0.0f, 1.0f);
    const float offset = phase < .64f ? std::lerp(stride, -stride, phase / .64f)
                                      : std::lerp(-stride, stride, smooth(swing));
    const auto floor = rig.point({x, 0.0f, z + offset});
    const float surface = (height(floor.x, floor.z) - rig.pose.origin.y) / rig.pose.scale;
    Vec3 hoof{x, surface + radius * .42f + std::sin(swing * pi) * .23f * leg_length * moving, z + offset};
    const auto displacement = hoof - hip;
    const float span = std::max(length(displacement), .0001f);
    const float upper = leg_length * .59f, lower = leg_length * .53f;
    const auto down = displacement * (1.0f / span);
    const float reach = std::clamp(span, .015f, upper + lower - .0001f);
    const float along = (reach * reach + upper * upper - lower * lower) / (2.0f * reach);
    const float bend = std::sqrt(std::max(0.0f, upper * upper - along * along));
    const auto flex = unit(forward - down * dot(forward, down)) * (rear ? 1.0f : -1.0f);
    const Vec3 knee = hip + down * along + flex * bend;
    rig.tube(hip, knee, radius * 1.5f, radius * .83f, rig.coat, pattern);
    rig.ball(knee, {radius * .84f, radius * .88f, radius * .85f}, rig.coat);
    rig.tube(knee, hoof, radius * .78f, radius * .49f, mix(rig.coat, dark, .20f), pattern);
    rig.ball(hoof + forward * (radius * .65f), {radius * .78f, radius * .58f, radius * 1.30f}, dark);
}

void quadruped(Rig& rig, const Species& species, AnimalForm form, const GroundHeight& height) {
    const auto shape = quadruped_shape(form);
    const float phase = rig.pose.phase, feeding = rig.pose.feeding;
    const float bob = std::sin(phase * tau * 2.0f) * .016f * rig.pose.moving + .004f * std::sin(phase * 1.7f);
    const float hip = rig.pose.swimming ? -.025f : shape.hip + bob;
    const float front_z = shape.trunk * .64f, rear_z = -shape.trunk * .68f;
    const auto terrain_at = [&](float x, float z) {
        if (rig.pose.swimming) return 0.0f;
        const auto p = rig.point({x, 0, z});
        return std::clamp((height(p.x, p.z) - rig.pose.origin.y) / rig.pose.scale, -.34f, .34f);
    };
    const float front = terrain_at(0, front_z), rear = terrain_at(0, rear_z);
    const auto body_axis = unit({0.0f, front - rear + (form == Hyena ? .12f : 0.0f), front_z - rear_z});
    const float body_y = hip + shape.belly * .62f + (front + rear) * .5f;
    const bool spotted = species.name == "Cheetah" || species.name == "Leopard" ||
        species.name == "Jaguar" || species.name == "Ocelot" || species.name == "Spotted hyena";
    const float pattern = form == Zebra || species.name == "Tiger" ? 1.0f : form == Giraffe ? 3.0f : spotted ? 2.0f : 0.0f;
    rig.ball({0, body_y, 0}, {shape.width, shape.belly, shape.trunk}, rig.coat, body_axis, pattern);
    rig.ball({0, body_y + .027f + front * .15f, front_z * .72f},
             {shape.width * .95f, shape.belly * .98f, shape.trunk * .56f}, rig.coat, body_axis, pattern);
    if (form == Bison || form == Camel) {
        rig.ball({0, body_y + .13f, form == Bison ? .16f : -.10f},
                 {shape.width * .78f, .19f, .21f}, mix(rig.coat, rig.accent, .14f));
    }
    const float toe_z = shape.head_z + (form == Giraffe ? .35f : .17f);
    const bool grazer = species.group == AnimalGroup::Grazer;
    const float jaw_y = std::lerp(shape.head_y + front + bob, terrain_at(0, toe_z) + (grazer ? .18f : .22f), feeding);
    const float jaw_z = std::lerp(shape.head_z, toe_z, feeding);
    const Vec3 neck_base{0.0f, body_y + .037f, front_z};
    const Vec3 head{0.0f, jaw_y, jaw_z};
    const float neck = shape.width * (form == Giraffe ? .51f : .66f);
    rig.tube(neck_base, head, neck * 1.14f, neck * .58f, rig.coat, pattern);
    if (form == Horse || form == Zebra) {
        const auto base = neck_base + Vec3{0,.105f,-.12f}, tip = head + Vec3{0,.072f,-.052f};
        rig.tube(base, tip, .065f, .051f, rig.accent);
    }
    const float head_radius = std::max(.068f, shape.width * .61f);
    const auto face = unit({0.0f, -.16f - feeding * .38f, 1.0f});
    rig.ball(head, {head_radius, head_radius * .92f, shape.muzzle * .79f}, rig.coat, face, pattern);
    const auto nose = head + face * (shape.muzzle * .72f);
    rig.ball(nose, {head_radius * .69f, head_radius * .48f, shape.muzzle * .53f},
             mix(rig.coat, rig.accent, .36f), face);
    rig.ball(nose + face * (shape.muzzle * .39f), {head_radius * .44f, head_radius * .28f, .030f}, dark);
    for (const float side : {-1.0f, 1.0f}) {
        const Vec3 eye = head + Vec3{side * head_radius * .82f, head_radius * .32f, shape.muzzle * .22f};
        rig.ball(eye, {.012f, .014f, .015f}, dark);
        const Vec3 ear = head + Vec3{side * head_radius * .83f, head_radius * .74f, -.057f};
        const bool long_ear = form == Ibex || form == Deer || form == Canid || form == Horse || form == Zebra;
        rig.ball(ear, {head_radius * (form == Elephant ? .91f : .38f),
                       head_radius * (form == Elephant ? 1.68f : long_ear ? 1.16f : .58f),
                       head_radius * (form == Elephant ? .30f : .44f)}, mix(rig.coat, rig.accent, .22f));
        if (form == Ibex || form == Deer || form == Bison || form == Ungulate || form == Giraffe) {
            const auto root = head + Vec3{side * head_radius * .49f, head_radius * .82f, -.035f};
            const float horn_height = form == Ibex ? .41f : form == Deer ? .28f : form == Bison ? .13f : .10f;
            rig.horn(root, side, horn_height, form == Ibex ? .42f : .16f, form == Ibex ? .057f : .040f);
            if (form == Deer) {
                const Vec3 fork = root + Vec3{side * .086f, .20f, -.075f};
                rig.tube(fork, fork + Vec3{side * .058f, .13f, .056f}, .029f, .002f, rig.accent);
                rig.tube(fork + up * .06f, fork + Vec3{side * .06f, .18f, -.02f}, .024f, .003f, rig.accent);
            }
        }
        if (form == Elephant) {
            const auto tusk = nose + Vec3{side * .075f, -.050f, .025f};
            const auto bend = tusk + Vec3{side * .028f, -.060f, .12f};
            rig.tube(tusk, bend, .038f, .024f, ivory);
            rig.tube(bend, bend + Vec3{side * .031f, .023f, .10f}, .025f, .002f, ivory);
        }
    }
    if (form == Elephant) {
        const Vec3 middle = nose + Vec3{.025f * std::sin(phase), -.21f, .095f};
        rig.tube(nose, middle, .088f, .056f, rig.coat);
        rig.tube(middle, middle + Vec3{.03f, -.22f + feeding * .07f, .085f}, .057f, .030f, rig.coat);
    } else if (form == Rhino) {
        rig.tube(nose + up * .04f, nose + Vec3{0.0f, .23f, .04f}, .073f, .002f, ivory);
    } else if (form == Lion) {
        rig.ball(head - forward * .085f, {head_radius * 1.55f, head_radius * 1.80f, .139f}, rig.accent);
    }
    const auto tail_root = Vec3{0, body_y + .02f, -shape.trunk * .86f};
    const auto tail_end = tail_root + Vec3{.092f * std::sin(phase * 3.2f), -shape.tail * .44f, -shape.tail};
    const auto tail_mid = mix(tail_root, tail_end, .52f) + up * .045f;
    rig.tube(tail_root, tail_mid, shape.shin * 1.07f, shape.shin * .66f, rig.coat);
    rig.tube(tail_mid, tail_end, shape.shin * .66f, .008f, mix(rig.coat, rig.accent, .53f));
    for (const bool behind : {false, true}) {
        for (const float side : {-1.0f, 1.0f}) {
            const float x = side * shape.width * .68f, z = behind ? rear_z : front_z;
            if (rig.pose.swimming) {
                const float kick = std::sin(phase * tau + side + (behind ? 1.8f : 0.0f));
                rig.blade({x, -.026f, z}, {x * 1.89f, -.086f, z - .15f + kick * .082f},
                          {x * .96f, -.074f, z - .21f}, .035f, rig.coat);
                continue;
            }
            const float y = hip + terrain_at(x, z) - (behind && form == Hyena ? .086f : 0.0f);
            const float offset = (side > 0.0f ? .50f : 0.0f) + (behind ? .73f : 0.0f);
            foot(rig, {x, y, z}, x, z, fractional(phase + offset), shape.hip, shape.shin, behind, height, pattern);
        }
    }
}

void bird(Rig& rig, AnimalForm form, std::string_view name, const GroundHeight& height) {
    const bool wader = form == Wader || form == Shorebird;
    const bool upright = form == Penguin || form == Ratite;
    const bool bat = form == Bat;
    const float air = rig.pose.airborne, phase = rig.pose.phase;
    const float long_legs = wader ? .51f : form == Ratite ? .45f : .21f;
    const float body_y = std::lerp(rig.pose.swimming ? .08f : long_legs + .14f, .22f, air);
    const auto axis = upright ? Vec3{0, .83f, .56f} : forward;
    rig.ball({0, body_y, 0}, {bat ? .22f : .18f, .151f, .34f}, rig.coat, axis);
    rig.ball({0, body_y - .064f, .046f}, {.156f, .114f, .272f}, mix(rig.coat, ivory, .30f), axis);
    const float head_y = body_y + (wader ? .39f : upright ? .33f : form == Waterfowl ? .24f : .15f)
        - rig.pose.feeding * (long_legs * .65f);
    const Vec3 neck{0, body_y + .044f, .225f}, head{0, head_y, .33f + rig.pose.feeding * .14f};
    const float head_r = form == Owl ? .143f : .093f;
    rig.tube(neck, head, wader ? .060f : .084f, .054f, rig.coat);
    const auto head_color = form == Waterfowl || form == Vulture || form == Owl ? rig.accent : rig.coat;
    rig.ball(head, {head_r, head_r * .93f, head_r * 1.07f}, head_color);
    const float bill = bat ? .075f : wader ? .40f : form == Hornbill ? .35f : form == Waterfowl ? .16f : .17f;
    const auto beak = head + Vec3{0, -.025f - rig.pose.feeding * .21f, bill};
    const bool stork = name == "White stork" || name == "Oriental stork";
    rig.tube(head + forward * .069f, beak, form == Hornbill ? .075f : .043f, .007f,
             bat ? rig.coat : stork ? rgb(0xb56355) : rig.accent);
    if (form == Hornbill) rig.ball(head + Vec3{0,.060f,.122f}, {.051f,.050f,.122f}, rig.accent);
    rig.blade({0, body_y, -.21f}, {-.109f, body_y + .01f, -.53f}, {.109f, body_y + .01f, -.53f}, .017f, rig.accent);
    for (const float side : {-1.0f, 1.0f}) {
        rig.ball(head + Vec3{side * head_r * .79f, .033f, .058f}, {.015f, .014f, .015f}, dark);
        if (bat) rig.ball(head + Vec3{side * .071f, .091f, -.034f}, {.048f,.091f,.028f}, rig.coat);
        const float spread = wader || form == Raptor || form == Vulture || bat ? .91f : .65f;
        const float flap = std::sin(phase * tau) * .29f * air;
        const auto wing_root = Vec3{side * .12f, body_y + .028f, .092f};
        const Vec3 wing_tip{side * std::lerp(.20f, spread, air), body_y + flap, std::lerp(-.35f, -.19f, air)};
        const Vec3 trailing{side * std::lerp(.16f, spread * .56f, air), body_y - .045f + flap * .35f, -.43f};
        if (air < .15f && !bat) {
            rig.ball(wing_root + Vec3{side * .044f, -.021f, -.17f}, {.080f,.078f,.26f}, mix(rig.coat, rig.accent, .43f));
        } else {
            rig.blade(wing_root, wing_tip, trailing, .043f, mix(rig.coat, rig.accent, .36f));
            if (!bat && air > .5f) {
                for (int feather = 0; feather < 4; ++feather) {
                    const float t = feather / 4.0f;
                    const auto edge = mix(wing_tip, trailing, t);
                    rig.blade(mix(wing_root, edge, .67f), edge + Vec3{side * .055f, -.021f, -.14f}, edge, .011f,
                              mix(rig.coat, rig.accent, .48f + t * .12f));
                }
            }
        }
        if (!rig.pose.swimming && !bat) {
            const Vec3 leg{side * .094f, body_y - .11f, -.033f};
            if (air < .18f) {
                const auto body_color = rig.coat;
                if (stork) rig.coat = rgb(0xb0635c);
                foot(rig, leg, leg.x, leg.z, fractional(phase + (side > 0 ? .5f : 0.0f)), long_legs, .025f, false, height);
                rig.coat = body_color;
            } else {
                rig.tube(leg, {leg.x, body_y - .15f, -.27f}, .024f, .015f, rig.accent);
            }
        }
    }
}

void wallaby(Rig& rig, const GroundHeight& height) {
    const auto pose = rig.pose;
    const GroundHeight hopping = [&](float x, float z) {
        return height(x, z) + pose.scale * .15f * std::max(0.0f, std::sin(pose.phase * tau)) * pose.moving;
    };
    rig.ball({0,.39f,-.08f}, {.17f,.25f,.25f}, rig.coat, {0,.76f,.65f});
    rig.ball({0,.57f,.08f}, {.113f,.21f,.13f}, rig.coat, {0,.80f,.60f});
    const Vec3 head{0, .81f - .32f * pose.feeding, .20f + .19f * pose.feeding};
    rig.tube({0,.55f,.10f}, head, .087f, .048f, rig.coat);
    rig.ball(head, {.084f,.088f,.121f}, rig.coat);
    rig.ball(head + forward * .104f, {.051f,.042f,.068f}, rig.accent);
    rig.tube({0,.31f,-.27f}, {0,.21f,-.63f}, .079f,.039f, rig.coat);
    rig.tube({0,.21f,-.63f}, {.055f,.11f,-.95f}, .04f,.006f, rig.coat);
    for (const float side : {-1.0f,1.0f}) {
        rig.ball(head + Vec3{side*.057f,.122f,-.051f}, {.037f,.107f,.034f}, rig.accent);
        rig.ball(head + Vec3{side*.072f,.019f,.061f}, {.015f,.015f,.017f}, dark);
        rig.tube({side*.10f,.58f,.13f}, {side*.13f,.36f,.31f}, .046f,.022f, rig.coat);
        foot(rig, {side*.128f,.37f,-.16f}, side*.134f, -.028f, fractional(pose.phase), .37f, .054f, true, hopping);
    }
}

void swimmer(Rig& rig, AnimalForm form, std::string_view name, const GroundHeight& height) {
    const float phase = rig.pose.phase * tau;
    const float wave = std::sin(phase) * (.048f + .14f * rig.pose.moving);
    if (form == Ray) {
        rig.ball({0,0,.075f}, {.265f,.081f,.342f}, rig.coat);
        for (const float side : {-1.0f,1.0f}) {
            const Vec3 root{side*.16f,.023f,.30f}, tip{side*.80f,wave,-.10f}, edge{side*.19f,-.025f,-.36f};
            rig.blade(root, tip, edge, .046f, rig.coat);
            rig.tube({side*.091f,-.01f,.33f}, {side*.132f,.013f,.51f}, .038f,.024f, rig.coat);
            rig.ball({side*.192f,.024f,.275f}, {.021f,.019f,.022f}, dark);
        }
        rig.tube({0,0,-.20f}, {wave*.31f,-.022f,-.53f}, .051f,.017f, rig.coat);
        rig.tube({wave*.31f,-.022f,-.53f}, {wave*.62f,0,-.98f}, .019f,.002f, rig.coat);
        return;
    }
    if (form == Krill) {
        for (int i = 0; i < 6; ++i) {
            const float z = .28f - i * .138f, t = 1.0f - i * .105f;
            rig.ball({0,-.052f * i + .025f * std::sin(phase + i),z}, {.12f*t,.12f*t,.14f}, mix(rig.coat, rig.accent, .10f * i));
        }
        for (const float side : {-1.0f,1.0f}) {
            rig.ball({side*.081f,.062f,.34f}, {.042f,.041f,.04f}, dark);
            rig.tube({side*.042f,.055f,.34f}, {side*.21f,.045f,.68f}, .015f,.002f, rig.accent);
            for (int i = 0; i < 4; ++i) {
                rig.tube({side*.06f,-.08f,.24f-i*.105f}, {side*.14f,-.23f,.18f-i*.105f + .032f * std::sin(phase+i)},
                         .022f,.004f, rig.coat);
            }
        }
        return;
    }
    const bool seal = form == Seal, shelled = form == SeaTurtle || form == Tortoise;
    const bool marine_mammal = form == Whale || form == Dolphin || seal;
    const float y = shelled && !rig.pose.swimming ? .20f : 0.0f;
    if (shelled) {
        rig.ball({0, y + .05f, -.03f}, {.31f, .215f, .409f}, rig.accent);
        rig.ball({0, y - .02f, -.015f}, {.306f, .109f, .407f}, rig.coat);
        const float front = .44f + .045f * std::sin(phase);
        const float head_y = y - rig.pose.feeding * .105f;
        rig.tube({0,y,.29f}, {0,head_y + .012f,front}, .086f,.055f, rig.coat);
        rig.ball({0,head_y + .025f,front}, {.073f,.072f,.102f}, rig.coat);
        rig.ball({.056f,head_y + .05f,front + .036f}, {.014f,.015f,.016f}, dark);
        rig.ball({-.056f,head_y + .05f,front + .036f}, {.014f,.015f,.016f}, dark);
        for (const float side : {-1.0f, 1.0f}) {
            for (const float z : {-.23f, .23f}) {
                if (rig.pose.swimming) {
                    rig.blade({side*.19f,y-.02f,z}, {side*(z > 0 ? .52f : .41f),y+wave,z-.22f},
                              {side*.21f,y-.037f,z-.23f}, .052f, rig.coat);
                } else {
                    foot(rig, {side*.22f,y,z}, side*.31f, z, fractional(rig.pose.phase + (side > 0 ? .45f : .0f) + z),
                         .23f, .059f, z < 0, height);
                }
            }
        }
        return;
    }
    const bool cetacean = form == Whale || form == Dolphin;
    const auto torso = Vec3{.185f, seal ? .181f : .163f, .37f};
    rig.ball({0, 0, .05f}, torso, rig.coat);
    rig.ball({0, -.066f, .09f}, {torso.x * .88f, torso.y * .65f, .31f}, mix(rig.coat, rig.accent, .65f));
    const auto head = Vec3{0, seal ? .055f : 0.0f, .345f};
    rig.ball(head, {torso.x * (form == Whale ? .93f : .65f), torso.y * .78f, seal ? .177f : .21f}, rig.coat);
    if (form == Dolphin && name != "Orca" && name != "Beluga whale") {
        rig.ball({0,-.027f,.51f}, {.084f,.047f,.13f}, rig.coat);
    }
    for (const float side : {-1.0f,1.0f}) {
        rig.ball({side*torso.x*.58f, .04f,.421f}, {.018f,.018f,.019f}, dark);
        const float span = form == Whale ? .34f : .29f;
        rig.blade({side * torso.x * .66f, -.03f, .24f}, {side * span, -.15f + wave * .25f, -.13f},
                  {side * torso.x * .9f, -.045f, -.30f}, .041f, rig.coat);
    }
    const auto tail_base = Vec3{0,-.019f,-.24f};
    const Vec3 tail{cetacean ? 0.0f : wave * .74f, cetacean ? wave : wave * .24f, -.56f};
    rig.tube(tail_base, tail, .115f, .027f, rig.coat);
    if (marine_mammal) {
        rig.blade(tail, tail + Vec3{.23f,.025f,-.136f}, tail + Vec3{.025f,.0f,-.19f}, .029f, rig.coat);
        rig.blade(tail, tail + Vec3{-.23f,.025f,-.136f}, tail + Vec3{-.025f,.0f,-.19f}, .029f, rig.coat);
    } else {
        rig.blade(tail, tail + Vec3{wave*.42f,.21f,-.11f}, tail + Vec3{wave*.42f,-.155f,-.13f}, .032f, rig.coat);
    }
    if (form == Shark || form == Dolphin || form == Fish) {
        rig.blade({0,.12f,.11f}, {0,form == Shark ? .37f : .25f,-.065f}, {0,.106f,-.26f}, .036f, rig.coat);
    }
}

} // namespace

void build_animal(Mesh& mesh, const Species& species, AnimalProfile profile, AnimalPose pose, const GroundHeight& height) {
    const float s = std::sin(pose.yaw), c = std::cos(pose.yaw);
    const auto coat = rgb(profile.coat);
    Rig rig{mesh, pose, {c, 0.0f, -s}, {s, 0.0f, c}, coat,
            profile.accent ? rgb(profile.accent) : mix(coat, ivory, .45f)};
    const auto motion = locomotion(profile.form);
    if (motion == Locomotion::Swim || profile.form == Tortoise) {
        swimmer(rig, profile.form, species.name, height);
    } else if (motion == Locomotion::Flight || motion == Locomotion::WaterBird || profile.form == Ratite) {
        bird(rig, profile.form, species.name, height);
    } else if (profile.form == Wallaby) {
        wallaby(rig, height);
    } else {
        quadruped(rig, species, profile.form, height);
    }
}

} // namespace wildlife::local
