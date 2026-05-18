library(tidyverse)
library(dplyr)

files <- list.files(path = "./input", pattern = "*.txt", full.names = TRUE)
names(files) <- basename(files)

df <- map(files, ~{
  read_delim(.x, delim = " ", col_names = FALSE) |> # read to dataframe
    select(X2, X4, X5) |> # only need the color, number of pumps, and if popped
    setNames(c("Color", "Pumps", "Is_Popped")) |> # set column names
    filter(Is_Popped != 1) |> # remove popped balloons
    mutate(Pumps = case_when( # calculate relative risk
      Color == "yellow" ~ Pumps / 31, # possible pumps is from 1-32 ie 31 values
      Color == "purple" ~ Pumps / 31,
      Color == "blue"   ~ Pumps / 127, # possible pumps is from 1-128
      Color == "orange" ~ Pumps / 127,
    )) |>
    select(Pumps) |> # only need relative risk scores now
    summarise(Risk = mean(Pumps)) # collapse to a single average
}) |>
  bind_rows(.id = "Sample") |>
  mutate(Sample = str_remove(Sample, "Data.txt$")) |>
  complete(Sample = c(Sample, # add all other samples
    paste0(str_remove(Sample, "BART"), "Break1"),
    paste0(str_remove(Sample, "BART"), "Colors"),
    paste0(str_remove(Sample, "BART"), "NBack"),
    paste0(str_remove(Sample, "BART"), "Break2"),
    paste0(str_remove(Sample, "BART"), "RC1"),
    paste0(str_remove(Sample, "BART"), "RC2"),
    paste0(str_remove(Sample, "BART"), "RC3")
  )) |>
  mutate(Condition = str_detect(Sample, "_H_")) |> # encode condition 1 or 0
  pivot_longer(cols = c(Condition, Risk)) |> # transpose
  pivot_wider(names_from = Sample) |>
  rename(Score = name)

write_csv(df, "./export/encodings.csv")
