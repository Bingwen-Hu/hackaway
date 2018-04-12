ExUnit.start

defmodule User do
  defstruct email: nil, password: nil
end

# impl in Rust
defimpl String.Chars, for: User do
  def to_string(%User{email: email}) do
    email
  end
end

defmodule RecordTest do
  use ExUnit.Case

  defmodule ScopeTest do
    use ExUnit.Case

    require Record
    Record.defrecordp :person, first_name: nil, last_name: nil, age: nil

    test "defrecordp" do
      # note: "str" != 'str'
      p = person(first_name: 'Kai', last_name: 'Morgan', age: 5)
      assert p == {:person, 'Kai', 'Morgan', 5}
    end
  end

  # test "defrecordp out of scope" do
  #   person()
  # end
  # >>> ** (CompileError) 06-record.exs:30: undefined function person/0

  def sample do
    # struct literals
    %User{email: "kai@example.com", password: "trains"}
  end

  test "defstruct" do
    assert sample == %{__struct__: User, email: "kai@example.com", password: "trains"}
  end

  # property syntax
  test "property" do
    assert sample.email == "kai@example.com"
  end

  # pattern matching update
  test "update" do
    u = sample
    u2 = %User{u | email: "tim@example.com"}
    assert u2 == %User{email: "tim@example.com", password: "trains"}
  end

  # traits in Rust, special method in Python
  test "protocol" do
    assert to_string(sample) == "kai@example.com"
  end
end
