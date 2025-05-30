use bevy::{prelude::*, tasks::futures_lite};
use client::InputManager;
use lightyear::prelude::*;

use crate::{
    ControlViaGrpc,
    protocol::{Direction, Inputs, PlayerPosition},
};

pub struct PlayerInputPlugin;

impl Plugin for PlayerInputPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, handle_input);
        app.add_systems(Update, sync_positions);
    }
}

fn handle_input(
    tick_manager: Res<TickManager>,
    mut input_manager: ResMut<InputManager<Inputs>>,
    keypress: Res<ButtonInput<KeyCode>>,
    grpc: Res<ControlViaGrpc>,
) {
    if grpc.enabled {
        let tick = tick_manager.tick();
        let grpc_input =
            futures_lite::future::block_on(async { grpc.current_input.lock().await.clone() });
        input_manager.add_input(grpc_input, tick)
    } else {
        let tick = tick_manager.tick();
        let mut input = Inputs::None;
        let mut direction = Direction {
            forward: false,
            back: false,
            left: false,
            right: false,
            reset: false,
        };
        if keypress.pressed(KeyCode::KeyW) || keypress.pressed(KeyCode::ArrowUp) {
            direction.forward = true;
        }
        if keypress.pressed(KeyCode::KeyS) || keypress.pressed(KeyCode::ArrowDown) {
            direction.back = true;
        }
        if keypress.pressed(KeyCode::KeyA) || keypress.pressed(KeyCode::ArrowLeft) {
            direction.left = true;
        }
        if keypress.pressed(KeyCode::KeyD) || keypress.pressed(KeyCode::ArrowRight) {
            direction.right = true;
        }
        if keypress.pressed(KeyCode::Space) {
            direction.reset = true;
        }
        if direction.is_some() {
            input = Inputs::Direction(direction);
        }
        input_manager.add_input(input, tick)
    }
}

fn sync_positions(mut players: Query<(&mut Transform, &PlayerPosition)>) {
    for (mut transform, position) in players.iter_mut() {
        *transform = transform
            .with_translation(position.0)
            .with_rotation(position.1);
    }
}
