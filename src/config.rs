use std::time::Duration;

use lightyear::prelude::*;

pub fn shared_config() -> SharedConfig {
    SharedConfig {
        server_replication_send_interval: Duration::from_millis(40),
        tick: TickConfig {
            tick_duration: Duration::from_secs_f64(1.0 / 64.0),
        },
        mode: Mode::Separate,
    }
}
