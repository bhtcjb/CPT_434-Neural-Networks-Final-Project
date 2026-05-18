library(tidyverse)
library(dplyr)

# set up file paths
eeg_files <- list.files(path = "./input/EEGfNIRS",
                        pattern = "eeg_processed.csv$",
                        full.names = TRUE)
names(eeg_files) <- basename(eeg_files)

fnirs_files <- list.files(path = "./input/EEGfNIRS",
                          pattern = "fNIRS_processed.csv$",
                          full.names = TRUE)
names(fnirs_files) <- basename(fnirs_files)

encodings <- read.csv("input/encodings/encodings.csv")



### Dataframe Formating Functions ###

read_eeg <- function(eeg_files) {

  # Starting format ... Channel|Band1|Band2|Band3|etc.
  df <- map(eeg_files, ~ { # for each file
    read_csv(.x, show_col_types = FALSE) |> # create dataframe

      # Format ... Channel|Band|Value
      pivot_longer(cols = -Channel) |> # flatten to column-wise labeled vector
      unite("Label", Channel, name, sep = "_") |> # combine y and x labels

      # Format ... Channel+Band1|Channel+Band2|Channel+Band3|etc.
      pivot_wider(names_from = Label) # transpose apply combined labels
  }) |>
    # Format ... Sample|Channel+Band1|Channel+Band2|Channel+Band3|etc.
    bind_rows(.id = "Sample") |> # label from each file
    mutate(Sample = str_remove(Sample, "_eeg_processed.csv$"),
           Sample = str_remove(Sample, "MUSE_"))
}

read_fnirs <- function(fnirs_files) {

  # Starting format ... Channel|Hemo
  df <- map(fnirs_files, ~ { # for each file
    read_csv(.x, show_col_types = FALSE) |> # create dataframe

      # Format ... Channel1|Channel2|Channel3|etc.
      pivot_wider(names_from = Channel, values_from = Hemo_Mean) # transpose
  }) |>
    # Format ... Sample|Channel1|Channel2|Channel3|etc.
    bind_rows(.id = "Sample") |> # label from each file
    mutate(Sample = str_remove(Sample, "_fNIRS_processed.csv$"),
           Sample = str_remove(Sample, "MUSE_"))
}

read_encodings <- function(encodings_file) {
  # Starting format ... Encodings|Sample1|Sample2|Sample3|etc.
  df <- encodings_file |>
    # Format ... Encodings|Sample|Value
    pivot_longer(cols = -Score, names_to = "Sample") |> # flatten

    # Format ... Sample|Encodings1|Encodings2
    pivot_wider(names_from = Score) # transpose
}

### Main ###

# create and format the dataframe
df <- read_eeg(eeg_files)
df <- left_join(df, read_fnirs(fnirs_files), by = "Sample")

# z-score normalization across all columns
df <- df |> mutate(across(where(is.numeric), ~ as.vector(scale(.x))))

# add encodings
df <- left_join(read_encodings(encodings), df, by = "Sample")

# remove rows with missing EEG or fNIRS data
df <- df |> drop_na(-Sample, -Risk)

print(df, width = Inf)
write_csv(df, "./export/Training_Data.csv")