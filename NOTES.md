I'm exploring a theory that a machine learning model could be trained to find patterns, and places where patterns break.  These two things would lead the model to all interesting discoveries.  I have a theory that this is what humans do.  That is why repetition is fun at first, but then gets boring, you are interested in the place where the pattern breaks.  To find that you first need to find the pattern.

This can be extended to all things we do.  Walking outside down a path, you could go for miles that all look the same, but you come to a stream crossing and you investigate.  From that most basic thing all the way to the abstract field of math.  It is all patterns.

It is why we like music.  Music is patterns, and breaking patterns.  You enjoy listening to the parts that repeat, and you enjoy the places where they change something.

For humans, patterns are one of the first things you can probably remember from when you were a little kid.  Learning to clap your hands in some kind of rythm.  The patterns of your daily life, and the time the pattern was broken and you got to go somewhere new.

A model could be trained on patterns, explaining patterns, finding the next item in a pattern, finding where a pattern breaks.  These can start with simple number patterns, but they could go to much larger concepts.  For example, you examine a script in a codebase, all things are done with modules creating very clean code, but one area is made up of dozens of lines of inline code that could be a module.  This inline code is a break in the pattern.  In writing you could have several paragraphs in a row that are written in third person, and then one that is written in first person.  This is a break in the pattern. 

Simple examples of pattern data:

instruction: Explain the pattern
input: 1, 2, 3, 4
output: The integer numbers in order


instruction: Find the next item in the pattern
input: 1, 2, 3, 4
output: 5


instruction: Where does the pattern break and why
input: 1, 2, 3, 4, 6
output: The pattern is integer numbers in order.  The 6 should be a 5.


instruction: Where does the pattern break and why
input: 1, 2, 4, 5, 6
output: The pattern is integer numbers in order.  3 is skipped.
