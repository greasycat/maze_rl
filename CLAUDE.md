# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Unity 6 (6000.3.6f1) project for a maze reinforcement learning environment. Currently using the URP Empty Template as a foundation.

## Build and Run

This is a standard Unity project without custom build scripts. All operations use Unity Editor:

- **Open project**: Use Unity Hub or Unity Editor 6000.3.6f1+
- **Play mode**: Ctrl+P in Unity Editor
- **Build**: File → Build Settings → Build
- **Run tests**: Window → General → Test Runner

## Architecture

### Rendering
- Universal Render Pipeline (URP) 17.3.0
- Separate renderers for Mobile (`Assets/Settings/Mobile_Renderer.asset`) and PC (`Assets/Settings/PC_Renderer.asset`)
- Linear color space

### Input System
- Uses Unity's New Input System (1.18.0)
- Input actions defined in `Assets/InputSystem_Actions.inputactions`
- Player action map: Move, Look, Attack, Interact, Crouch, Jump, Sprint, Previous, Next
- UI action map: Navigate, Submit, Cancel, Point, Click, ScrollWheel
- Control schemes: Keyboard&Mouse, Gamepad, Touch, Joystick, XR

### Key Directories
- `Assets/Scenes/` - Game scenes (SampleScene.unity is the main scene)
- `Assets/Settings/` - URP renderers and volume profiles
- `ProjectSettings/` - Unity project configuration

### Dependencies
Key packages (see `Packages/manifest.json`):
- `com.unity.ai.navigation` (2.0.9) - AI navigation/pathfinding
- `com.unity.inputsystem` (1.18.0) - Input handling
- `com.unity.render-pipelines.universal` (17.3.0) - Graphics
- `com.unity.test-framework` (1.6.0) - Testing
