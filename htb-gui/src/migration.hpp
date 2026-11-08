#pragma once

#include "globe.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace wildlife {

class MigrationAtlas {
public:
    struct Species {
        std::string name;
        ImU32 color;
        std::size_t animals = 0;
    };

    struct Year {
        int number;
        double epoch;
        double begin;
        double end;
    };

    struct Observation {
        std::size_t animal;
        std::size_t next_fix;
        Vec3 position;
        float speed_kmh;
    };

    struct Location {
        Vec3 position;
        double time;
        std::size_t animal;
    };

    explicit MigrationAtlas(const std::filesystem::path& executable = {});

    bool empty() const { return tracks_.empty(); }
    const std::vector<Species>& species() const { return species_; }
    const std::vector<Year>& years() const { return years_; }
    std::size_t animal_count() const { return tracks_.size(); }
    std::size_t animal_species(std::size_t index) const { return tracks_[index].species; }
    const std::string& animal_name(std::size_t index) const { return tracks_[index].name; }
    std::optional<Location> locate_species(std::size_t species, double time) const;

    void sample(double time, std::span<const std::uint8_t> enabled, std::vector<Observation>& out) const;
    std::optional<std::size_t> draw(ImDrawList& draw, const GlobeView& view, double time,
        std::span<const Observation> observations, double trail_hours, std::optional<std::size_t> selected,
        std::optional<ImVec2> pointer) const;

private:
    struct Fix {
        std::int64_t time;
        Vec3 position;
        bool joins_previous;
    };

    struct Track {
        std::string name;
        std::size_t species;
        std::vector<Fix> fixes;
    };

    bool load(const std::filesystem::path& path);
    static std::size_t sample_track(const Track& track, double time, Vec3& position);
    static void draw_trail(ImDrawList& draw, const GlobeView& view, const Track& track,
                           const Observation& observation, double time, double trail_hours, ImU32 color);

    std::vector<Species> species_;
    std::vector<Track> tracks_;
    std::vector<Year> years_;
};

float track_distance_km(Vec3 a, Vec3 b);

} // namespace wildlife
