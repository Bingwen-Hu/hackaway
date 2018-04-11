# mory note
list = ~w(Jenny Ann Mory)
for name <- list do
  IO.puts(name)
end


for name <- list, String.first(name) != "M" do
  IO.puts(name)
end

surpass = IO.gets("what's your surpass kind? ")
IO.puts(surpass)
case String.downcase surpass |> String.trim do
  "wind" -> IO.puts("Wow!")
  "fire" -> IO.puts("Orz..")
  _ -> IO.puts("#{surpass} Nothing special...")
end
