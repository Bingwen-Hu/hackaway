
// this is comment like C
val lst = List("Mory", "Ann", "Jenny")

lst.foreach(s => s.toUpperCase)

for (name <- lst)
  println(name)

// a tuple
val tuple = ("Mory", 1, 'a')
print(tuple._1)
println(tuple)
