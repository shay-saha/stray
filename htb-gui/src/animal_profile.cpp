#include "simulation.hpp"

#include <cctype>
#include <initializer_list>
#include <string>

namespace wildlife {

Locomotion locomotion(AnimalForm form) {
    using enum AnimalForm;
    switch (form) {
        case Seal: case Whale: case Dolphin: case Fish: case Shark: case Ray: case SeaTurtle: case Krill:
            return Locomotion::Swim;
        case Penguin: case Waterfowl: case Seabird:
            return Locomotion::WaterBird;
        case Bat: case Wader: case Shorebird: case Raptor: case Vulture: case Hornbill: case Songbird: case Owl:
            return Locomotion::Flight;
        default:
            return Locomotion::Ground;
    }
}

float maximum_grade(AnimalForm form) {
    using enum AnimalForm;
    switch (form) {
        case Ibex: return 1.04f;
        case Primate: case Canid: case Felid: return .67f;
        case Tortoise: case Elephant: case Rhino: case Crocodile: return .32f;
        case Wader: case Waterfowl: case Penguin: return .36f;
        default: return .52f;
    }
}

AnimalProfile animal_profile(const Species& species) {
    if (species.appearance.length > 0.0f) return species.appearance;

    // The atlas's remaining species use the closest locomotion and body plan.
    using enum AnimalForm;
    std::string name(species.name);
    for (auto& letter : name) letter = static_cast<char>(std::tolower(static_cast<unsigned char>(letter)));
    const auto has = [&](std::initializer_list<std::string_view> parts) {
        for (const auto part : parts) {
            if (name.find(part) != std::string::npos) return true;
        }
        return false;
    };
    switch (species.group) {
        case AnimalGroup::Bird:
            if (has({"penguin"})) return {Penguin, has({"emperor", "king"}) ? 1.05f : .70f, 0x393f47, 0xe0d9c7};
            if (has({"ostrich", "emu", "rhea", "cassowary"})) return {Ratite, 1.45f, 0x77706a, 0x9d8266};
            if (has({"stork", "crane", "flamingo", "spoonbill", "ibis", "shoebill", "heron"})) {
                return {Wader, 1.03f, has({"flamingo"}) ? 0xd7ada4u : 0xb7b7aeu, 0x574c48};
            }
            if (has({"eagle", "falcon", "gyrfalcon", "hawk", "kestrel"})) return {Raptor, .68f, 0x72604a, 0xb39972};
            if (has({"vulture", "condor"})) return {Vulture, 1.03f, 0x504a44, 0xb19b85};
            if (has({"owl"})) return {Owl, .48f, 0xb9b6aa, 0x93806f};
            if (has({"goose", "swan", "duck"})) return {Waterfowl, .84f, 0xa6a49b, 0x7f654e};
            if (has({"petrel", "albatross", "pelican", "gull", "auk", "booby", "gannet"})) {
                return {Seabird, has({"albatross", "pelican"}) ? 1.20f : .48f, 0xa6aaab, 0x555962};
            }
            if (has({"hornbill", "toucan"})) return {Hornbill, .64f, 0x4b4642, 0xcd9852};
            if (has({"macaw", "parrot", "kakapo", "cockatoo", "kākāpō"})) {
                return {has({"kakapo", "kākāpō"}) ? Ratite : Songbird, .58f,
                    has({"scarlet", "red-and"}) ? 0xaa5748u : has({"blue", "hyacinth"}) ? 0x516c91u : 0x71895cu, 0xc49a59};
            }
            if (has({"kiwi", "grouse", "bustard", "ptarmigan", "prairie-chicken"})) return {Ratite, .53f, 0x827568, 0xaf9b84};
            return {Songbird, .30f, 0x807c63, 0xcbb485};
        case AnimalGroup::Grazer:
            if (has({"elephant"})) return {Elephant, 5.10f, 0x796f65, 0xbca490};
            if (has({"rhinoceros"})) return {Rhino, 3.40f, 0x82796d, 0xaea69a};
            if (has({"giraffe"})) return {Giraffe, 3.80f, 0xbda276, 0x856041};
            if (has({"zebra"})) return {Zebra, 2.25f, 0xd3cec1, 0x393638};
            if (has({"horse", "kiang", "ass"})) return {Horse, 2.25f, 0x9a7955, 0x46382c};
            if (has({"camel", "vicuna", "guanaco"})) return {Camel, has({"camel"}) ? 2.9f : 1.65f, 0xb29268, 0xd7c6a6};
            if (has({"ibex", "chamois", "tahr", "markhor", "bharal", "sheep", "mountain goat", "serow", "goral"})) {
                return {Ibex, 1.34f, 0x8a6f54, 0x514638};
            }
            if (has({"deer", "pudu", "wapiti", "caribou"})) return {Deer, has({"pudu"}) ? .75f : 1.65f, 0x937353, 0xc0a482};
            if (has({"bison", "buffalo", "yak", "gaur", "takin", "anoa", "banteng", "muskox", "wildebeest"})) {
                return {Bison, 2.45f, 0x685448, 0x9b8971};
            }
            if (has({"tapir", "hippopotamus"})) return {Tapir, has({"hippopotamus"}) ? 3.10f : 1.85f, 0x6d635e, 0xb2a596};
            if (has({"peccary"})) return {Boar, .99f, 0x6a5546, 0xa69378};
            if (has({"mara", "prairie dog", "capybara"})) return {SmallMammal, has({"capybara"}) ? 1.1f : .55f, 0x9b815f, 0xb6a38c};
            return {Ungulate, 1.45f, 0xa78a66, 0xd3bd94};
        case AnimalGroup::LandCarnivore:
            if (has({"bear", "panda"})) return {Bear, has({"polar", "brown"}) ? 2.40f : 1.65f,
                has({"polar", "panda"}) ? 0xb9b6a8u : 0x68574au, has({"panda"}) ? 0x35323au : 0x9d8f7bu};
            if (has({"lion"})) return {Lion, 1.92f, 0xb79861, 0x6e4e35};
            if (has({"hyena"})) return {Hyena, 1.32f, 0x937955, 0x544632};
            if (has({"wolf", "fox", "dog", "dingo", "dhole", "coyote"})) {
                return {Canid, has({"fox"}) ? .71f : 1.24f, has({"fox"}) ? 0xa87952u : 0x77726au, 0xc2b6a1};
            }
            if (has({"badger", "ferret", "marten", "mink", "otter", "wolverine", "meerkat", "mongoose"})) {
                return {SmallMammal, .71f, 0x6c5b4d, 0xb39b7e};
            }
            return {Felid, has({"tiger"}) ? 2.27f : has({"leopard"}) ? 1.44f : .88f,
                has({"snow leopard"}) ? 0xb0aca3u : 0xae865b, 0x554237};
        case AnimalGroup::Primate:
            return {Primate, has({"gorilla"}) ? 1.35f : has({"tamarin", "squirrel"}) ? .34f : .71f, 0x64554b, 0x998575};
        case AnimalGroup::AquaticMammal:
            if (has({"seal", "walrus", "sea lion"})) return {Seal, has({"walrus"}) ? 3.25f : 2.23f, 0x7d7770, 0xa9a197};
            if (has({"dolphin", "porpoise", "narwhal", "beluga", "orca", "killer whale", "pilot whale"})) {
                return {Dolphin, has({"orca", "killer whale", "pilot whale"}) ? 5.8f : 2.55f,
                    has({"beluga"}) ? 0xc6c8c4u : 0x53606b, 0xbfc5c4};
            }
            return {Whale, has({"blue whale", "fin whale"}) ? 23.0f : 13.0f, 0x545d69, 0x9aa6ac};
        case AnimalGroup::FishAndKrill:
            if (has({"krill"})) return {Krill, .058f, 0xb8887a, 0xe1ad98};
            if (has({"ray"})) return {Ray, 3.75f, 0x54636c, 0xa9b7b9};
            if (has({"shark", "mako", "hammerhead"})) return {Shark, has({"whale", "basking"}) ? 9.0f : 3.45f, 0x606b75, 0xb9bfc0};
            return {Fish, has({"tuna", "arapaima"}) ? 1.45f : .39f, 0x64757c, 0xa0afad};
        case AnimalGroup::Reptile:
            if (has({"sea turtle"})) return {SeaTurtle, 1.0f, 0x65684f, 0xab9979};
            if (has({"tortoise"})) return {Tortoise, .60f, 0x6a6652, 0xac9c7f};
            if (has({"crocodile", "alligator", "gharial"})) return {Crocodile, 3.60f, 0x676b50, 0xa9a185};
            return {Lizard, .75f, 0x6d7157, 0xae9a76};
        case AnimalGroup::OtherMammal:
            if (has({"kangaroo", "wallaby", "wallaroo", "quokka"})) return {Wallaby, 1.12f, 0x89755f, 0xbcaa91};
            if (has({"bat", "flying fox"})) return {Bat, .25f, 0x5e4d43, 0x8d7565};
            if (has({"boar", "pig", "hog", "babirusa"})) return {Boar, 1.30f, 0x69584a, 0xa1917d};
            return {SmallMammal, has({"anteater", "armadillo"}) ? 1.0f : .51f, 0x837360, 0xb4a38a};
        case AnimalGroup::Count: break;
    }
    return {Ungulate, 1.2f, 0x917d65, 0xc0aa8f};
}

} // namespace wildlife
