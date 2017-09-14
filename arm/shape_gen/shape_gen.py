from Tkinter import *
import turtle 
import random

def randColor():
	color = [0]*3
	for i in range(3):
		color[i] = random.randint(1,255)
	return tuple(color)

def randCoord(size):
	x = size[0]/2
	y = size[0]/2
	return (random.randint(10-x, x-10), random.randint(10-y, y-10))

def drawBackground(ttl, size=(64,64)):
	ttl.up()
	ttl.setpos(size)
	ttl.down()
	c = randColor()
	ttl.pencolor(c)
	ttl.fillcolor(c)
	ttl.begin_fill()
	ttl.goto((size[0], -size[1]))
	ttl.goto((-size[0], -size[1]))
	ttl.goto((-size[0], size[1]))
	ttl.goto(size)
	ttl.end_fill()

def drawShape(ttl, sides=3, size=(64,64)):
	initCoord = randCoord(size)
	ttl.up()
	ttl.setpos(initCoord)
	ttl.down()
	c = randColor()
	ttl.pencolor(c)
	ttl.fillcolor(c)
	ttl.begin_fill()
	for i in range(sides-1):
		ttl.goto(randCoord(size))
	ttl.goto(initCoord)
	ttl.end_fill()

N = 100000
size = (128, 128)

turtle.setup(size[0], size[1])
canvas = turtle.Screen()
turtle.colormode(255)
ttl = turtle.Turtle()
ttl.hideturtle()
ttl.speed(0)

for i in range(N):
	if i % 100 == 0:
		print i
	ttl.clear()
	drawBackground(ttl, size)
	drawShape(ttl, random.randint(3,5), size)
	ts = ttl.getscreen()
	ts.getcanvas().postscript(file="quad/%d.eps"%(i))

canvas.exitonclick()
