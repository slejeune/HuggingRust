use std::path::{Path, PathBuf};

use crate::{hub::ImportError, models::ModelMetadata};

pub(super) async fn existing_files<'a>(
    dir: &Path,
    files: &[&'a str],
    metadata: &ModelMetadata,
    pb: &indicatif::ProgressBar,
) -> Result<(Vec<PathBuf>, Vec<(&'a str, PathBuf)>), ImportError> {
    let mut paths = Vec::new();
    let mut missing = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for file in files {
        let path = destination(dir, file)?;
        if !seen.insert(path.clone()) {
            pb.finish_with_message(format!("✗ {} has colliding file names", metadata.model));
            return Err(ImportError::FileCollision { path });
        }
        if is_valid_file(&path).await {
            paths.push(path);
        } else {
            missing.push((*file, path));
        }
    }
    Ok((paths, missing))
}

pub(super) async fn install_file(
    cached: &Path,
    destination: &Path,
    model: &str,
    file: &str,
) -> Result<(), ImportError> {
    if let Some(parent) = destination.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|source| ImportError::CreateDir {
                path: parent.to_path_buf(),
                source,
            })?;
    }
    let filename = destination
        .file_name()
        .ok_or_else(|| ImportError::InvalidFilename {
            model: model.to_owned(),
            file: file.to_owned(),
        })?
        .to_string_lossy();
    let temp = destination.with_file_name(format!("{filename}.download-part"));
    let result = async {
        tokio::fs::copy(cached, &temp).await?;
        if !is_valid_file(&temp).await {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "downloaded file is empty or not a regular file",
            ));
        }
        tokio::fs::rename(&temp, destination).await
    }
    .await;
    if let Err(source) = result {
        let _ = tokio::fs::remove_file(&temp).await;
        return Err(ImportError::CopyFile {
            model: model.to_owned(),
            file: file.to_owned(),
            destination: destination.to_path_buf(),
            source,
        });
    }
    Ok(())
}

fn destination(dir: &Path, file: &str) -> Result<PathBuf, ImportError> {
    let relative = Path::new(file);
    let is_safe = !relative.as_os_str().is_empty()
        && !relative.is_absolute()
        && relative
            .components()
            .all(|component| matches!(component, std::path::Component::Normal(_)));
    is_safe
        .then(|| dir.join(relative))
        .ok_or_else(|| ImportError::InvalidFilename {
            model: dir.display().to_string(),
            file: file.to_owned(),
        })
}

async fn is_valid_file(path: &Path) -> bool {
    match tokio::fs::metadata(path).await {
        Ok(metadata) => metadata.is_file() && metadata.len() > 0,
        Err(_) => false,
    }
}
