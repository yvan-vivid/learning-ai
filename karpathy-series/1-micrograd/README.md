# Yvan Vivid's Micrograd

This is my interpretation of the ["micrograd" library](https://github.com/karpathy/micrograd) that Karpathy created for learning about back propagation. I wanted to try and construct something that worked out the idea but with a couple additional properties:

1. Typed.
2. The expression graph is generic and can be manipulated without knowing what is in it.
3. The graph is acyclic, immutable, and topologically ordered by construction.
4. The forward and back propagation of values are not entangled in the graph data structure.
5. There were some nice notational properties I think I managed to achieve.

I will continue to work on this a bit more, but this was really just an exercise in thinking through how I would try to organize this kind of thing.
