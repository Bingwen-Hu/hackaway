val filesHere = (new java.io.File(".")).listFiles

// print out
for (
  file <- filesHere
  if file.isFile
  if file.getName.endsWith(".scala")
) println(file)


// collect
def files = for (
  file <- filesHere
  if file.isFile
  if file.getName.endsWith(".scala")
) yield file

// nested
def fileLines(file: java.io.File) = 
  scala.io.Source.fromFile(file).getLines().toList

def grep(pattern: String) = 
  for {
    file <- filesHere
    if file.getName.endsWith(".scala")
    line <- fileLines(file)
    trimmed = line.trim
    if trimmed.matches(pattern)
  } println(file + ": " + trimmed)

// very power!

