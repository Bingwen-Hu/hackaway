ExUnit.start

defmodule ListTest do
  use ExUnit.Case

  def sample do
    ["Tim", "Jen", "Mac", "Kai"]
  end

  test "sigil" do
    assert sample == ~w(Tim Jen Mac Kai)
  end

  test "head" do
    [head | _] = sample
    assert head == "Tim"
  end

  test "tail" do
    [_ | tail] = sample
    assert tail == ~w(Jen Mac Kai)
  end

  # warning, has to traverse the entire list
  # means: not effective
  test "last item" do
    assert List.last(sample) == "Kai"
  end

  test "delete item" do
    assert List.delete(sample, "Mac") == ~w(Tim Jen Kai)
    # only deletes the first occurrence
    assert List.delete([1, 2, 2, 3], 2) == [1, 2, 3]
    # not change at all!
    assert sample == ~w(Tim Jen Mac Kai)
  end

  test "List.fold" do
    list = [20, 10, 5, 2.5]
    sum = List.foldr list, 0, &(&1 + &2) # lambda expression
    assert sum == 37.5
    sum = List.foldr list, 0, fn (num, sum) -> num + sum end
    assert sum == 37.5
  end

  # preferred
  test "Enum.reduce" do
    list = [20, 10, 5, 2.5]
    sum = Enum.reduce list, 0, &(&1 + &2)
    assert sum == 37.5
  end

  # like list in Python
  test "wrap" do
    assert List.wrap(sample) == sample
    assert List.wrap(1) == [1]
    assert List.wrap([]) == []
    assert List.wrap(nil) == []
  end

  test "list comprehension" do
    some = for n <- sample, String.first(n) < "M", do: n <> " Morgan"
    assert some == ["Jen Morgan", "Kai Morgan"]
  end

  # reverse list
  test "manual reverse speed" do
    {microsec, reversed} = :timer.tc fn ->
      Enum.reduce 1..1_000_000, [], fn (i, l) -> List.insert_at(l, 0, i) end
    end
    assert reversed == Enum.to_list(1_000_000..1)
    IO.puts "manual reverse took #{microsec} microsecs"
  end

  # expensive to insert to the end of a list
  test "speed of inserting at the end of a list" do
    {microsec, reversed} = :timer.tc fn ->
      Enum.reduce 1..10000, [], fn (i, l) -> List.insert_at(l, -1, i) end
    end
    assert reversed == Enum.to_list(1..10000)
    IO.puts "Inserting at the end of a list took #{microsec} microsecs"
  end

  test "Enum.reverse speed" do
    {microsec, reversed} = :timer.tc fn ->
      Enum.reverse 1..1_000_000
    end
    assert reversed == Enum.to_list(1_000_000..1)
    IO.puts "Enum.reverse took #{microsec} microsecs"
  end


end
