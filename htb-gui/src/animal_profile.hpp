#pragma once

#include <cstdint>

namespace wildlife {

enum class AnimalForm {
    Ungulate, Ibex, Deer, Horse, Zebra, Bison, Elephant, Rhino, Giraffe, Camel, Boar, Tapir,
    Canid, Felid, Lion, Bear, Hyena, SmallMammal, Primate, Wallaby,
    Tortoise, Crocodile, Lizard, Ratite, Penguin,
    Bat, Waterfowl, Wader, Shorebird, Raptor, Vulture, Seabird, Hornbill, Songbird, Owl,
    Seal, Whale, Dolphin, Fish, Shark, Ray, SeaTurtle, Krill
};

struct AnimalProfile {
    AnimalForm form = AnimalForm::Ungulate;
    float length = 0.0f;
    std::uint32_t coat = 0x8e7358;
    std::uint32_t accent = 0;
};

enum class Locomotion { Ground, Flight, WaterBird, Swim };

struct Species;
AnimalProfile animal_profile(const Species& species);
Locomotion locomotion(AnimalForm form);
float maximum_grade(AnimalForm form);

} // namespace wildlife
