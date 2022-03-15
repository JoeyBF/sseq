use std::{sync::Once, thread, time::Instant};

use dashmap::DashMap;
use itertools::Itertools;
use once_cell::sync::Lazy;
use signal_hook::{
    consts::{SIGTSTP, SIGUSR1},
    iterator::Signals,
    low_level::emulate_default_handler,
};

static CURRENT_TASKS: Lazy<DashMap<isize, Task>> = Lazy::new(DashMap::new);
static INIT: Once = Once::new();

#[non_exhaustive]
pub enum Task {
    Idle,
    StepResolution(Instant, u32, i32),
}

impl Default for Task {
    fn default() -> Self {
        Task::Idle
    }
}

impl ToString for Task {
    fn to_string(&self) -> String {
        match self {
            Task::Idle => "Idle".to_string(),
            Task::StepResolution(start, s, t) => {
                let duration = start.elapsed();
                format!(
                    "({:>6}.{:>06} s) Computing bidegree ({n}, {s})",
                    duration.as_secs(),
                    duration.subsec_micros(),
                    n = t - *s as i32,
                    s = s
                )
            }
        }
    }
}

fn initialize() {
    INIT.call_once(|| {
        let mut signals =
            Signals::new(&[SIGTSTP, SIGUSR1]).expect("Failed to register signal handler");

        thread::Builder::new()
            .name("signal handling".to_string())
            .spawn(move || {
                for sig in signals.forever() {
                    log_current_tasks();
                    if sig == SIGTSTP {
                        emulate_default_handler(sig)
                            .expect("Failed to call default signal handler");
                    }
                }
            })
            .expect("Failed to spawn signal handling thread");
    });
}

pub fn log_current_tasks() {
    initialize();
    let tasks = CURRENT_TASKS
        .iter()
        .sorted_by_key(|x| *x.key())
        .collect_vec();
    for pair in tasks {
        let (id, task) = pair.pair();
        if *id == -1 {
            eprintln!("Main thread: {str_task}", str_task = task.to_string());
        } else {
            eprintln!("Thread {id:<4}: {str_task}", str_task = task.to_string());
        }
    }
}

pub fn set_current_task(task: Task) -> Task {
    initialize();
    let id = rayon::current_thread_index()
        .map(|i| i as isize)
        .unwrap_or(-1);
    CURRENT_TASKS.insert(id, task).unwrap_or_default()
}
