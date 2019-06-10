# Classification 

In this example, I will classify mushrooms as being edible or poisonous depending on different features. keras will be used.

The data set contains 8124 rows and the following features:

class: edible(e) or poisonous(p)

cap-shape: bell(b), conical(c), convex(x), flat(f), knobbed(k), sunken(s)

cap-surface: fibrous(f), grooves(g), scaly(y), smooth(s)

cap-color: brown(n), buff(b), cinnamon(c), gray(g), green(r), pink(p), purple(u), red(e), white(w), yellow(y)

bruises: bruises(t), no bruises(f)

odor: almond(a), anise(l), creosote(c), fishy(y), foul(f), musty(m), none(n), pungent(p), spicy(s)

gill-attachment: attached(a), descending(d), free(f), notched(n)

gill-spacing: close(c), crowded(w), distant(d)

gill-size: broad(b), narrow(n)

gill-color: black(k), brown(n), buff(b), chocolate(h), gray(g), green(r), orange(o), pink(p), purple(u), red(e), white(w), yellow(y)

stalk-shape: enlarging(e), tapering(t)

stalk-root: bulbous(b), club(c), cup(u), equal(e), rhizomorphs(z), rooted(r), missing(?)

stalk-surface-above-ring: fibrous(f), scaly(y), silky(k), smooth(s)

stalk-surface-below-ring: fibrous(f), scaly(y), silky(k), smooth(s)

stalk-color-above-ring: brown(n), buff(b), cinnamon(c), gray(g), orange(o), pink(p), red(e), white(w), yellow(y)

stalk-color-below-ring: brown(n), buff(b), cinnamon(c), gray(g), orange(o), pink(p), red(e), white(w), yellow(y)

veil-type: partial(p), universal(u)

veil-color: brown(n), orange(o), white(w), yellow(y)

ring-number: none(n), one(o), two(t)

ring-type: cobwebby(c), evanescent(e), flaring(f), large(l), none(n), pendant(p), sheathing(s), zone(z)

spore-print-color: black(k), brown(n), buff(b), chocolate(h), green(r), orange(o), purple(u), white(w), yellow(y)

population: abundant(a), clustered(c), numerous(n), scattered(s), several(v), solitary(y)

habitat: grasses(g), leaves(l), meadows(m), paths(p), urban(u), waste(w), woods(d)


## Development server

The server.py has been implemented in how to encode data and training data with ML methods.
You can use the Anaconda application to run the Python files on Windows.
To run the server.py in the Anaconda environment, you first need to install the following packages by executing the following commands in Anaconda’s terminal.
[Keras, Tensorflow, Flask] 

`pip install tensorflow`
`pip install flask`
`pip install keras`

After installing the above packages, you can run the file server by running the following command. 
`Python server.py`
