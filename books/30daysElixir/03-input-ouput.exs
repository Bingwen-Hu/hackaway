defmodule CowInterrogator do

  # docstring
  @doc """
  Get name from standard IO
  """

  # input function
  def get_name do
    IO.gets("what is your name? ")
    |> String.trim
  end

  # specifi input length I think!
  def get_cow_lover do
    IO.getn("Do you like cows? [yes|no] ", 3)
  end

  def interrogate do
    name = get_name
    case String.downcase(get_cow_lover) |> String.trim do
      "yes" ->
        IO.puts "Great! Here's a cow for you #{name}:"
        IO.puts cow_art
      "no" ->
        IO.puts "That's a shame, #{name}."
      _ ->
        IO.puts "You should have entered 'y' or 'n'."
    end
  end

  def cow_art do
    path = Path.expand("cow.txt", __DIR__)
    case File.read(path) do
      {:ok, art} -> art
      {:error, _} -> IO.puts "Error: cow.txt file not found"; System.halt(1)
    end
  end
end

ExUnit.start

defmodule InputOutputTest do
  use ExUnit.Case
  import String

  test "checks if cow_art returns string from support/cow.txt" do
    # this call checks if cow_art function returns art from txt file
    art = CowInterrogator.cow_art
    assert trim(art) |> first == "(" # String.first
  end
end


CowInterrogator.interrogate
