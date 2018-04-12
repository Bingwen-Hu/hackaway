ExUnit.start

defmodule MapTest do
  use ExUnit.Case

  def sample do
    %{foo: 'bar', bar: 'quz'}
  end

  test "Map.get" do
    assert Map.get(sample, :foo) == 'bar'
    assert Map.get(sample, :non_existent) == nil
  end

  # like get in Python.dict
  test "[]" do
    assert sample[:foo] == 'bar'
    assert sample[:non_existent] == nil
  end

  # only for :keywords
  test "." do
    assert sample.foo == 'bar'
    assert_raise KeyError, fn ->
      sample.non_existent
    end
  end

  # wrap in Option
  test "Map.fetch" do
    {:ok, val} = Map.fetch(sample, :foo)
    assert val == 'bar'
    :error = Map.fetch(sample, :non_existent)
  end

  test "Map.put" do
    # updating existing key
    assert Map.put(sample, :foo, 'bob') == %{foo: 'bob', bar: 'quz'}
    assert Map.put(sample, :far, 'bar') == %{foo: 'bar', bar: 'quz', far: 'bar'}
  end

  test "Update map using pattern matching syntax" do
    assert %{sample | foo: 'bob'} == %{foo: 'bob', bar: 'quz'}
    # could not use to insert
    assert_raise KeyError, fn ->
      %{sample | far: 'bob'}
    end
  end

  test "Map.values" do
    assert Enum.sort(Map.values(sample)) == ['bar', 'quz']
  end

end
