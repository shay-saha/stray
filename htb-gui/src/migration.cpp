#include "migration.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <numbers>
#include <string_view>
#include <utility>

namespace wildlife {
namespace {

constexpr int seconds_per_hour = 3600;
constexpr int max_gap = 24 * seconds_per_hour;
constexpr float max_speed_kmh = 200.0f;
constexpr std::array palette{
    IM_COL32(238, 176, 220, 255), IM_COL32(126, 233, 223, 255), IM_COL32(255, 205, 129, 255),
    IM_COL32(183, 175, 255, 255), IM_COL32(255, 155, 139, 255), IM_COL32(152, 218, 157, 255),
    IM_COL32(139, 196, 248, 255), IM_COL32(251, 239, 198, 255), IM_COL32(214, 153, 255, 255),
    IM_COL32(255, 177, 112, 255), IM_COL32(142, 233, 191, 255), IM_COL32(234, 150, 175, 255),
};

bool fields_of(std::string_view line, std::array<std::string_view, 5>& fields) {
    for (std::size_t i = 0; i + 1 < fields.size(); ++i) {
        const auto comma = line.find(',');
        if (comma == std::string_view::npos) return false;
        fields[i] = line.substr(0, comma);
        line.remove_prefix(comma + 1);
    }
    fields.back() = line;
    return !line.empty() && line.find(',') == std::string_view::npos;
}

bool parse_coordinate(std::string_view field, float limit, float& value) {
    char* end = nullptr;
    value = std::strtof(field.data(), &end);
    return !field.empty() && end == field.data() + field.size() && std::isfinite(value) && std::abs(value) <= limit;
}

Vec3 interpolate(Vec3 a, Vec3 b, float weight) {
    const Vec3 point{std::lerp(a.x, b.x, weight), std::lerp(a.y, b.y, weight), std::lerp(a.z, b.z, weight)};
    const float inverse = 1.0f / std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
    return {point.x * inverse, point.y * inverse, point.z * inverse};
}

ImU32 opacity(ImU32 color, float alpha) {
    return (color & ~IM_COL32_A_MASK) | (static_cast<ImU32>(alpha) << IM_COL32_A_SHIFT);
}

bool inside(ImVec2 point, ImVec2 min, ImVec2 max) {
    return point.x >= min.x && point.x <= max.x && point.y >= min.y && point.y <= max.y;
}

void glow_dot(ImDrawList& draw, ImVec2 center, ImU32 color, float strength) {
    constexpr int segments = 12;
    const float radius = 2.8f + 4.0f * strength;
    draw.PrimReserve(segments * 3, segments + 1);
    const auto base = draw._VtxCurrentIdx;
    const ImVec2 uv = ImGui::GetFontTexUvWhitePixel();
    draw.PrimWriteVtx(center, uv, opacity(color, 125.0f * strength));
    for (int i = 0; i < segments; ++i) {
        const float angle = static_cast<float>(i) * 2.0f * std::numbers::pi_v<float> / segments;
        draw.PrimWriteVtx({center.x + radius * std::cos(angle), center.y + radius * std::sin(angle)}, uv, opacity(color, 0));
        draw.PrimWriteIdx(static_cast<ImDrawIdx>(base));
        draw.PrimWriteIdx(static_cast<ImDrawIdx>(base + 1 + i));
        draw.PrimWriteIdx(static_cast<ImDrawIdx>(base + 1 + (i + 1) % segments));
    }
}

} // namespace

float track_distance_km(Vec3 a, Vec3 b) {
    const float chord = std::hypot(a.x - b.x, a.y - b.y, a.z - b.z);
    return 2.0f * 6371.0f * std::asin(std::min(1.0f, chord * 0.5f));
}

MigrationAtlas::MigrationAtlas(const std::filesystem::path& executable) {
    constexpr auto filename = "wildlife-tracks.csv";
    const auto bin = executable.parent_path();
    const std::array paths{bin / filename, bin.parent_path() / "Resources" / filename,
                           std::filesystem::path("assets") / filename};
    for (const auto& path : paths) {
        if (load(path)) return;
    }
}

bool MigrationAtlas::load(const std::filesystem::path& path) {
    using namespace std::chrono;
    std::ifstream file(path);
    std::string line;
    if (!std::getline(file, line)) return false;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line != "species,bird_id,unix_seconds,latitude,longitude") return false;

    std::vector<Species> species;
    std::vector<Track> tracks;
    std::map<int, Year> years;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        std::array<std::string_view, 5> fields;
        if (!fields_of(line, fields) || fields[0].empty() || fields[1].empty()) return false;
        std::int64_t time = 0;
        const auto time_end = fields[2].data() + fields[2].size();
        const auto parsed = std::from_chars(fields[2].data(), time_end, time);
        float latitude = 0.0f;
        float longitude = 0.0f;
        if (parsed.ec != std::errc{} || parsed.ptr != time_end || time < 0 || time > 4102444800LL ||
            !parse_coordinate(fields[3], 90.0f, latitude) || !parse_coordinate(fields[4], 180.0f, longitude)) return false;

        if (tracks.empty() || fields[0] != species.back().name || fields[1] != tracks.back().name) {
            const auto previous = tracks.empty() ? std::pair<std::string_view, std::string_view>{}
                : std::pair<std::string_view, std::string_view>{species.back().name, tracks.back().name};
            if (std::pair{fields[0], fields[1]} < previous) return false;
            if (species.empty() || fields[0] != species.back().name) {
                species.push_back({std::string(fields[0]), palette[species.size() % palette.size()]});
            }
            ++species.back().animals;
            tracks.push_back({std::string(fields[1]), species.size() - 1, {}});
        }
        auto& fixes = tracks.back().fixes;
        const Vec3 position = earth_position(latitude, longitude);
        bool joins_previous = false;
        if (!fixes.empty()) {
            const auto gap = time - fixes.back().time;
            if (gap <= 0) return false;
            joins_previous = gap <= max_gap &&
                track_distance_km(fixes.back().position, position) <= max_speed_kmh * gap / seconds_per_hour;
        }
        fixes.push_back({time, position, joins_previous});

        const year_month_day date{floor<days>(sys_seconds{seconds{time}})};
        const int year = static_cast<int>(date.year());
        const double epoch = duration_cast<seconds>(sys_days{date.year() / January / 1}.time_since_epoch()).count();
        auto& summary = years.try_emplace(year, Year{year, epoch, static_cast<double>(time), static_cast<double>(time)}).first->second;
        summary.begin = std::min(summary.begin, static_cast<double>(time));
        summary.end = std::max(summary.end, static_cast<double>(time));
    }
    if (file.bad() || tracks.empty()) return false;
    species_ = std::move(species);
    tracks_ = std::move(tracks);
    for (const auto& entry : years) years_.push_back(entry.second);
    return true;
}

std::size_t MigrationAtlas::sample_track(const Track& track, double time, Vec3& position) {
    const auto& fixes = track.fixes;
    const double rounded = std::round(time);
    if (std::abs(time - rounded) < 1e-6) time = rounded;
    if (time < fixes.front().time || time > fixes.back().time) return fixes.size();
    const auto next = std::lower_bound(fixes.begin(), fixes.end(), time,
        [](const Fix& fix, double cursor) { return fix.time < cursor; });
    if (next == fixes.end()) return fixes.size();
    if (next->time == time) {
        position = next->position;
    } else {
        if (!next->joins_previous) return fixes.size();
        const auto& previous = *(next - 1);
        const float fraction = static_cast<float>((time - previous.time) / (next->time - previous.time));
        position = interpolate(previous.position, next->position, fraction);
    }
    return static_cast<std::size_t>(next - fixes.begin());
}

void MigrationAtlas::sample(double time, std::span<const std::uint8_t> enabled, std::vector<Observation>& out) const {
    out.clear();
    for (std::size_t index = 0; index < tracks_.size(); ++index) {
        const auto& track = tracks_[index];
        if (!enabled[track.species]) continue;
        Vec3 position;
        const auto next = sample_track(track, time, position);
        if (next == track.fixes.size()) continue;
        const auto& fix = track.fixes[next];
        float speed = 0.0f;
        if (fix.joins_previous) {
            const auto& previous = track.fixes[next - 1];
            speed = track_distance_km(previous.position, fix.position) * seconds_per_hour / (fix.time - previous.time);
        }
        out.push_back({index, next, position, speed});
    }
}

std::optional<MigrationAtlas::Location> MigrationAtlas::locate_species(std::size_t species, double time) const {
    std::optional<Location> active;
    std::optional<Location> location;
    std::optional<Location> passage;
    double best_distance = std::numeric_limits<double>::infinity();
    double passage_distance = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < tracks_.size(); ++i) {
        const auto& track = tracks_[i];
        if (track.species != species) continue;
        Vec3 position;
        const auto sample = sample_track(track, time, position);
        if (sample != track.fixes.size()) {
            const Location current{position, time, i};
            if (track.fixes[sample].joins_previous && track_distance_km(track.fixes[sample - 1].position, position) >= 0.1f) {
                return current;
            }
            if (!active) active = current;
            continue;
        }
        if (active) continue;
        const auto next = std::lower_bound(track.fixes.begin(), track.fixes.end(), time,
            [](const Fix& fix, double cursor) { return fix.time < cursor; });
        const auto consider = [&](const Fix& fix) {
            const double distance = std::abs(fix.time - time);
            if (distance < best_distance) {
                best_distance = distance;
                location = Location{fix.position, static_cast<double>(fix.time), i};
            }
        };
        if (next != track.fixes.end()) consider(*next);
        if (next != track.fixes.begin()) consider(*(next - 1));
        for (std::size_t j = 1; j < track.fixes.size(); ++j) {
            const auto& a = track.fixes[j - 1];
            const auto& b = track.fixes[j];
            if (!b.joins_previous) continue;
            const double midpoint = 0.5 * (a.time + b.time);
            const double distance = std::abs(midpoint - time);
            if (distance < passage_distance && track_distance_km(a.position, b.position) >= 0.1f) {
                passage_distance = distance;
                passage = Location{interpolate(a.position, b.position, 0.5f), midpoint, i};
            }
        }
    }
    if (active) return active;
    return passage ? passage : location;
}

void MigrationAtlas::draw_trail(ImDrawList& draw, const GlobeView& view, const Track& track,
                                const Observation& observation, double time, double trail_hours, ImU32 color) {
    const double start = time - trail_hours * seconds_per_hour;
    const auto& fixes = track.fixes;
    const ImVec2 min = draw.GetClipRectMin();
    const ImVec2 max = draw.GetClipRectMax();
    const auto stroke = [&](ImVec2 a, ImVec2 b, double at) {
        if (std::max(a.x, b.x) < min.x || std::min(a.x, b.x) > max.x ||
            std::max(a.y, b.y) < min.y || std::min(a.y, b.y) > max.y) return;
        const float dx = b.x - a.x;
        const float dy = b.y - a.y;
        if (dx * dx + dy * dy < 0.04f) return;
        const float strength = static_cast<float>(std::clamp((at - start) / (time - start), 0.0, 1.0));
        draw.AddLine(a, b, opacity(color, 29.0f * strength), 4.8f);
        draw.AddLine(a, b, opacity(color, 182.0f * strength), 1.25f);
    };

    Vec3 newer_position = observation.position;
    double newer_time = time;
    ImVec2 newer_point{};
    bool newer_visible = view.project(newer_position, newer_point);
    for (std::size_t i = observation.next_fix; i > 0 && fixes[i].joins_previous && newer_time > start; --i) {
        const auto& older = fixes[i - 1];
        const bool reaches_start = older.time <= start;
        const auto position = reaches_start
            ? interpolate(older.position, newer_position, static_cast<float>((start - older.time) / (newer_time - older.time)))
            : older.position;
        ImVec2 point{};
        const bool visible = view.project(position, point);
        if (visible && newer_visible) stroke(point, newer_point, newer_time);
        if (reaches_start) break;
        newer_position = position;
        newer_time = older.time;
        newer_point = point;
        newer_visible = visible;
    }
}

std::optional<std::size_t> MigrationAtlas::draw(ImDrawList& draw, const GlobeView& view, double time,
    std::span<const Observation> observations, double trail_hours, std::optional<std::size_t> selected,
    std::optional<ImVec2> pointer) const {
    const ImVec2 min = draw.GetClipRectMin();
    const ImVec2 max = draw.GetClipRectMax();
    if (trail_hours > 0.0) {
        for (const auto& observation : observations) {
            const auto& track = tracks_[observation.animal];
            draw_trail(draw, view, track, observation, time, trail_hours, species_[track.species].color);
        }
    }

    std::optional<std::size_t> nearest;
    float best_distance = 9.0f * 9.0f;
    for (const auto& observation : observations) {
        ImVec2 point;
        float facing = 0.0f;
        if (!view.project(observation.position, point, &facing) || !inside(point, min, max)) continue;
        const auto color = species_[tracks_[observation.animal].species].color;
        const float travel = std::clamp(observation.speed_kmh / 35.0f, 0.0f, 1.0f);
        const float limb = std::clamp(facing * 6.0f, 0.2f, 1.0f);
        if (travel > 0.09f) glow_dot(draw, point, color, travel * limb);
        draw.AddCircleFilled(point, 1.1f + 0.6f * travel, opacity(color, (182.0f + 72.0f * travel) * limb), 7);
        if (selected == observation.animal) draw.AddCircle(point, 5.0f, opacity(color, 235), 16, 1.15f);
        if (pointer) {
            const float dx = pointer->x - point.x;
            const float dy = pointer->y - point.y;
            const float distance = dx * dx + dy * dy;
            if (distance < best_distance) {
                nearest = observation.animal;
                best_distance = distance;
            }
        }
    }
    return nearest;
}

} // namespace wildlife
