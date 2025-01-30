use std::{path::Path, sync::Arc, time::Duration};

use tokio::{
    fs::OpenOptions,
    io::{AsyncBufReadExt, BufReader},
    process::Command,
    sync::{broadcast, Semaphore},
};

const LOG_FILE: &str = "target/log_file.log";
const MAX_CONCURRENT_ZSTD: usize = 4;

#[tokio::main]
async fn main() {
    let semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_ZSTD));
    let (tx, _) = broadcast::channel::<String>(100);

    tokio::spawn(monitor_log(tx.clone()));

    let mut handles = vec![];
    for _ in 0..MAX_CONCURRENT_ZSTD {
        let semaphore = Arc::clone(&semaphore);
        let mut rx = tx.subscribe();

        let handle = tokio::spawn(async move {
            while let Ok(file_path) = rx.recv().await {
                compress_file(&file_path, &semaphore).await;
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.await.unwrap();
    }
}

async fn monitor_log(tx: broadcast::Sender<String>) {
    loop {
        match OpenOptions::new().read(true).open(LOG_FILE).await {
            Ok(file) => {
                let reader = BufReader::new(file);
                let mut lines = reader.lines();

                while let Ok(Some(line)) = lines.next_line().await {
                    if let Some(file_path) = extract_file_path(&line) {
                        let _ = tx.send(file_path); // Ignore send failures
                    }
                }
            }
            Err(_) => {
                println!("Log file not available, retrying...");
            }
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

fn extract_file_path(line: &str) -> Option<String> {
    let re = regex::Regex::new("closing file=\"(.+)\"").unwrap();
    re.captures(line).map(|caps| caps[1].to_string())
}

async fn compress_file(file_path: &str, semaphore: &Semaphore) {
    let _permit = semaphore.acquire().await.unwrap();

    if Path::new(file_path).exists() {
        println!("Compressing {}", file_path);
        let mut process = Command::new("zstd")
            .args(["-f", "--rm", "--ultra", "-22", "--verbose"])
            .arg(file_path)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("Failed to start zstd process");

        if let Some(stderr) = process.stderr.take() {
            let mut reader = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                println!("[ZSTD {}]: {}", file_path, line);
            }
        }

        let status = process
            .wait()
            .await
            .expect("Failed to wait on zstd process");

        if status.success() {
            println!("Finished compressing {}", file_path);
        } else {
            println!("Error compressing {}", file_path);
        }
    } else {
        println!("File not found: {}", file_path);
    }
}
