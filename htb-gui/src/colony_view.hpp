#pragma once

#include "simulation.hpp"

#include <imgui.h>
#include <memory>
#include <optional>

namespace wildlife {

class ColonyView {
public:
    ColonyView();
    ~ColonyView();
    bool open(const Simulation& simulation, AnimalId animal);
    void close();
    bool active() const;
    void toggle_playback();
    std::optional<AnimalId> draw(const Simulation& simulation, bool night);

private:
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace wildlife
