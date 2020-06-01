#pragma once

#include <tuple>

// FW Declares
struct SDL_Window;
using SDL_GLContext = void*;

namespace FM32
{

std::tuple<SDL_Window*, SDL_GLContext, const char*> InitSDL();



}