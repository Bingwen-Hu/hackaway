ExUnit.start

defmodule MyTest do
  use ExUnit.Case

  test 'simple test' do
    assert 1 + 1 == 2
  end

  test "refute is opposite of assert" do
    refute 1 + 1 == 3
  end

  test :assert_raise do
    assert_raise ArithmeticError, fn ->
      1 + "x"
    end
  end
end
