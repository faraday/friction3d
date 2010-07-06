SCREEN_SIZE = (800, 600)
HALF_SIZE = (SCREEN_SIZE[0]/2.0,SCREEN_SIZE[1]/2.0)

CG = 100.0
#G = 0.00001
G = 9.8

WORLD_LX = 100.0
WORLD_LY = 200.0

GOAL_DIST = 20

SCALE_FACTOR = WORLD_LY / WORLD_LX

FRICTION = 0.5
STOP_THRES = 0.0001

SPEED_INC = 50
TURN_INC = 0.05
ZOOM_INC = 2.0

RECT_TURN_INC = 2

SHOT_LIFE = 3

SHOT_RADIUS = 0.8

import threading
import time

import ode

from math import radians
import math 

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import pygame
from pygame.locals import *

from gameobjects import *
from gameobjects.matrix44 import *

import random

def resize(width, height):
    
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60.0, float(width)/height, .1, WORLD_LY*5)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def init():
    
    glEnable(GL_DEPTH_TEST)
    
    glShadeModel(GL_SMOOTH)
    glClearColor(0.0, 0.0, 0.0, 0.0)

    glColorMaterial ( GL_FRONT, GL_SPECULAR )
    glEnable(GL_COLOR_MATERIAL)
    
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLight(GL_LIGHT0, GL_POSITION,  (0, 1, 0, 0))	# old: (0,1,0,0)

    glLightfv(GL_LIGHT1, GL_AMBIENT, (0.2,0.2,0.2,1.0) )
    glLightfv(GL_LIGHT1, GL_DIFFUSE, (1.0,1.0,1.0,1.0) )
    glLightfv(GL_LIGHT1, GL_SPECULAR, (1.0,1.0,1.0,1.0) )
    glLightfv(GL_LIGHT1, GL_POSITION, (0,-1,0,0) )	# old: (0,-1,0,0)
    glLightfv(GL_LIGHT1, GL_AMBIENT, (1.0,1.0,1.0,1.0) )

    glLightfv(GL_LIGHT1, GL_CONSTANT_ATTENUATION, 1.5 )
    glLightfv(GL_LIGHT1, GL_LINEAR_ATTENUATION, 0.5 )
    glLightfv(GL_LIGHT1, GL_QUADRATIC_ATTENUATION, 0.2 )

    glEnable(GL_LIGHT1)


class Sphere(object):

    uid = -1
    body = None
    geom = None
    
    def __init__(self, position, color):
	self.type = 'ball'
        self.position = position
        self.color = color

	# set radius & density as constants for now
	self.radius = 0.8
	self.density = 1000

    def create_body(self,space,world):
	body = ode.Body(world)
	M = ode.Mass()
	M.setSphere(self.density,self.radius)
	body.setMass(M)
	body.setPosition(self.position)

	self.geom = ode.GeomSphere(space,self.radius)
	self.geom.setBody(body)

	self.body = body

    def set_radius(self,radius):
	self.radius = radius

    def set_position(self,position):
	self.position = position

    def set_type(self,type):
	self.type = type

    def set_material(self):	# type = ball / world ?
	if self.type == 'ball' or self.type == 'shot':
    		glMaterial(GL_FRONT, GL_AMBIENT, (0.0, 0.0, 0.0, 1.0))    
    		glMaterial(GL_FRONT, GL_DIFFUSE, (self.color[0], self.color[1], self.color[2], 1.0))
    		glMaterial(GL_FRONT, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))
    		glMaterial(GL_FRONT, GL_SHININESS, (50.0))
	elif self.type == 'world':
    		glMaterial(GL_FRONT, GL_AMBIENT, (0.3, 0.3, 0.3, 1.0))    
    		glMaterial(GL_FRONT, GL_DIFFUSE, (0.8, 0.8, 0.8, 1.0))

    def move(self,inc_vector):
	(x,y,z) = self.position
	(ix,iy,iz) = inc_vector
	self.position = (x + ix, y + iy, z + iz) 

    def draw(self):
        quadric = gluNewQuadric()
        glColor( self.color )
	gluQuadricOrientation(quadric,GLU_OUTSIDE)
	self.set_material()
	if self.type == 'world':
        	gluSphere(quadric,self.radius,100,50)
        	#glutWireSphere(self.radius,100,50)
	else:	# ball
        	gluSphere(quadric,self.radius,100,200)	# 20,10
        	#glutWireSphere(self.radius,10,20)
	gluDeleteQuadric(quadric)
    
    def render(self):
	if not self.body:
		glPushMatrix()
		glTranslatef(self.position[0],self.position[1],self.position[2])
		glCallList(self.uid)
		glPopMatrix()
		return
	(x,y,z) = self.body.getPosition()
	R = self.body.getRotation()
	rot = [R[0], R[3], R[6], 0.,
           R[1], R[4], R[7], 0.,
           R[2], R[5], R[8], 0.,
           x, y, z, 1.0]
    	glPushMatrix()
    	glMultMatrixd(rot)

	glCallList(self.uid)

	glPopMatrix()

    
class Box(object):

    uid = -1
    body = None
    geom = []
    norm = []
    
    def __init__(self, position, color):
	self.type = 'ball'
        self.position = position
        self.color = color

	# set radius & density as constants for now
	#self.radius = 0.8
	self.density = 1000

	self.lx = 1.0
	self.ly = 1.0
	self.lz = 1.0

    def create_body(self,space,world):
	'''
	body = ode.Body(world)
	M = ode.Mass()
	M.setBox(self.density,self.lx,self.ly,self.lz)
	body.setMass(M)
	body.setPosition(self.position)

	body.shape = "box"
	body.boxsize = (self.lx,self.ly,self.lz)
	'''

	#self.geom = ode.GeomBox(space,body.boxsize)
	#self.geom.setBody(body)

	self.norm.append((1,0,0))	
	self.norm.append((0,0,1))
	self.norm.append((-1,0,0))
	self.norm.append((0,0,-1))

	for n in self.norm:
		self.geom.append(ode.GeomPlane(space,n,-self.lx/2))


	ny1 = (0,1,0)
	ny2 = (0,-1,0)
	self.norm.append(ny1)
	self.norm.append(ny2)

	self.geom.append(ode.GeomPlane(space,ny1,-self.ly/2))
	self.geom.append(ode.GeomPlane(space,ny2,-self.ly/2))

	#self.geom.append(ode.GeomPlane(space,(0,0,1),-self.lx))


	for i in range(6):
		self.geom[i].type = 'plane'	

	'''
	for i in range(6):
		body = ode.Body(world)
		M = ode.Mass()
		M.setBox(self.density,0.5,0.5,0.5)
		body.setMass(M)
		body.setPosition(b[i])

		self.geom[i].setBody(body)
	'''


	#self.body = body

    def set_lx(self,lx):
	self.lx = lx

    def set_ly(self,ly):
	self.ly = ly

    def set_lz(self,lz):
	self.lz = lz

    def set_position(self,position):
	self.position = position

    def set_type(self,type):
	self.type = type

    def set_material(self):	# type = ball / world ?
	if self.type == 'ball' or self.type == 'shot':
    		glMaterial(GL_FRONT, GL_AMBIENT, (0.0, 0.0, 0.0, 1.0))    
    		glMaterial(GL_FRONT, GL_DIFFUSE, (self.color[0], self.color[1], self.color[2], 1.0))
    		glMaterial(GL_FRONT, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))
    		glMaterial(GL_FRONT, GL_SHININESS, (50.0))
	elif self.type == 'world':
    		glMaterial(GL_FRONT, GL_AMBIENT, (0.3, 0.3, 0.3, 1.0))    
    		glMaterial(GL_FRONT, GL_DIFFUSE, (0.8, 0.8, 0.8, 1.0))

    def move(self,inc_vector):
	(x,y,z) = self.position
	(ix,iy,iz) = inc_vector
	self.position = (x + ix, y + iy, z + iz) 

    # draw cube
    def draw_cube(self):
        glColor( self.color )
	self.set_material()
	if self.type == 'world':
        	glutWireCube(self.lx)
	else:	# ball
        	glutWireCube(self.lx)

    # draw prism
    def draw(self):
        glColor( self.color )
	self.set_material()
	if self.type == 'world':
		glScalef(1,SCALE_FACTOR,1)
        	#glutSolidCube(self.lx)
        	glutWireCube(self.lx)
	else:	# ball
		glScalef(1,SCALE_FACTOR,1)
        	#glutSolidCube(self.lx)
        	glutWireCube(self.lx)

    def render(self):
	if not self.body:
		glPushMatrix()
		glTranslatef(self.position[0],self.position[1],self.position[2])
		glCallList(self.uid)
		glPopMatrix()
		return
	(x,y,z) = self.body.getPosition()
	R = self.body.getRotation()
	rot = [R[0], R[3], R[6], 0.,
           R[1], R[4], R[7], 0.,
           R[2], R[5], R[8], 0.,
           x, y, z, 1.0]
    	glPushMatrix()
    	glMultMatrixd(rot)

	glCallList(self.uid)

	glPopMatrix()

class Shot(Sphere):
	direction = (0.0,0.0,0.0)
	speed = 0
	stopper = 100	# to stop, speed should be low for 20 consecutive steps
    
	def __init__(self, position, color):
		self.type = 'shot'
        	self.position = position
        	self.color = color

		# set radius & density as constants for now
		self.radius = SHOT_RADIUS
		self.density = 1000.0

	def set_direction(self,vector):
		direction = vector

	def set_speed(self,speed):
		self.speed = speed

def point_to_plane(point,planeNormal,d):
	(a,b,c) = planeNormal
	(x0,y0,z0) = point
	return abs(a*x0+b*y0+c*z0+d) / math.sqrt(a**2+b**2+c**2)

def from_spherical(rho,theta,phi):
    cam_x = rho * cos(theta) * sin(phi)
    cam_y = rho * cos(phi)
    cam_z = rho * sin(theta) * sin(phi)
    up_x = -cos(theta) * cos(phi)
    up_y = sin(phi)
    up_z = -sin(theta) * cos(phi)
    return (cam_x,cam_y,cam_z,up_x,up_y,up_z)

# dist : distance from origin in Y-axis
def from_rectangle(yabs,recx,recz):
    cam_x = recx
    cam_y = yabs
    cam_z = recz
    return (cam_x,cam_y,cam_z,0,0,-1)

def vector_mag(vector):
	return math.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)

def normalize_vector(vector):
	mag = vector_mag(vector)
	if not mag:
		return (0.0,0.0,0.0)
	return (vector[0]/mag,vector[1]/mag,vector[2]/mag)

def minus_vector(v1,v2):
	return (v1[0]-v2[0],v1[1]-v2[1],v1[2]-v2[2])

def add_vector(v1,v2):
	return (v1[0]+v2[0],v1[1]+v2[1],v1[2]+v2[2])

def invert_vector(v):
	return (-v[0],-v[1],-v[2])

# geometric utility functions
def scalp (vec, scal):
    return (vec[0] * scal, vec[1] * scal, vec[2] * scal)

def scald (vec, scal):
    return (vec[0] / scal, vec[1] / scal, vec[2] / scal)

def crossp(vec1,vec2):
    return  (vec1[1]*vec2[2] - vec1[2]*vec2[1],vec1[2]*vec2[0] - vec1[0]*vec2[2],vec1[0]*vec2[1] - vec1[1]*vec2[0])

def rotatev(vec1,axis,angle):
    #print angle * (180 / math.pi)
    s = math.sin(angle)
    c = math.cos(angle)
    u = 1 - c
    (x,y,z) = axis
    (vx,vy,vz) = vec1
    rx = (x**2*u + c) * vx + (y*x*u - z*s) * vy + (z*x*u + y*s) * vz
    ry = (x*y*u + z*s) * vx + (y**2*u + c) * vy + (z*y*u - x*s) * vz
    rz = (x*z*u - y*s) * vx + (y*z*u + x*s) * vy + (z**2*u + c) * vz
    #print rx, ry, rz
    return (rx,ry,rz)


curShot = -1
remShots = {}

# Collision callback
def near_callback(args, geom1, geom2):
    """Callback function for the collide() method.

    This function checks if the given geoms do collide and
    creates contact joints if they do.
    """

    global curShot, remShots

    # Check if the objects do collide
    contacts = ode.collide(geom1, geom2)

    if geom1.type == 'shot' and geom2.type == 'shot':
	#print 'col: ' , geom1.uid, geom2.uid
	ku = -1
	if geom1.uid == curShot:
		ku = geom2.uid
	elif geom2.uid == curShot:
		ku = geom1.uid

	dk = vector_mag(minus_vector(geom1.getBody().getPosition(),geom2.getBody().getPosition()))
	sumradius = geom1.getRadius() + geom2.getRadius()

	#print 'debug: ',sumradius,dk
	if ku != -1 and sumradius >= dk:
		if not remShots.has_key(ku):
			remShots[ku] = 0
		remShots[ku] += 1

		#print curShot, remShots

    # Create contact joints
    world,contactgroup = args
    for c in contacts:
        c.setBounce(1.0)
        c.setMu(5000)
        j = ode.ContactJoint(world, contactgroup, c)
        j.attach(geom1.getBody(), geom2.getBody())


class Game:
	PCAM_Y = WORLD_LY / 2

    	rho = 59.0	# the distance from the object that the camera is targeting
    	theta = 0.0 # The angle of rotation around the Z axis, where 0 degrees means parallel to the x axis
    	phi = 0.0	# the angle of declination from the Z axis - 0 degrees means you're directly overhead, 90 degrees means you're level with the object

	recx = 0
	recz = 0

	goal_y = WORLD_LY/2 + 1

	balls = []
	ball_area = 10.0
	cam_coords = (0.0,0.0,0.0)

	shot_speed = 100.0
	shot_vector = (0.0,0.0,0.0)
	shots = []

	allStopped = True
	process = False
	stopShots = []

	wasdown = False

	total_mass = 0.0

	com = (0.0,0.0,0.0)	# center of mass of ALL balls (including shots)

	t_step = 0	# sim time

	# ODE variables
	ode_world = ode.World()

	# game score
	score = 0

	net_uid = -1


	def __init__(self):
		# ODE settings
		self.ode_world.setERP(0.8)
		self.ode_world.setCFM(1E-5)

		self.space = ode.Space()
		#self.wall = ode.GeomSphere(self.space,50.0)
		self.contactgroup = ode.JointGroup()

		# mega box - in which we are in
    		self.world = Box((0.0,0.0,0.0),(1.0,1.0,1.0))
    		#self.world.set_radius(60.0)
		self.world.set_lx(WORLD_LX)
		self.world.set_ly(WORLD_LY)
		self.world.set_lz(WORLD_LX)
    		self.world.set_type('world')

		self.world.create_body(self.space,self.ode_world)

		self.world.uid = glGenLists(1)
		glNewList(self.world.uid,GL_COMPILE)
		self.world.draw()
		glEndList()


    		self.net_uid = glGenLists(1)
    		glNewList(self.net_uid,GL_COMPILE)
    		glMaterial(GL_FRONT, GL_AMBIENT, (0.2, 1.0, 0.2, 1.0))    
    		glMaterial(GL_FRONT, GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))
    		self.netLines(self.world.ly/2-GOAL_DIST,10)
    		glEndList()

		#self.generate_balls(50)

	def calculate_com(self):	# calculate center of mass
		count = 0.0
		for ball in self.balls:
			pos = ball.body.getPosition()
			m = ball.body.getMass().mass
			pos = scalp(pos,m)
			self.com = add_vector(self.com,pos)
			count += m

		self.com = scald(self.com,count)
		self.total_mass = count

		for shot in self.shots:
			vel =  shot.body.getLinearVel()
			pos = shot.body.getPosition()

			svel = vector_mag(vel)
			sdiff = vector_mag(minus_vector(self.com,pos))

			# shot is out of space boundaries
			became_ball = False
			if sdiff > 70.0:
				self.shots.remove(shot)
				#became_ball = True
			# shot becomes part of object
			elif sdiff < 10.0: # svel < 0.05 and sdiff < 10.0:
				#self.shots.remove(shot)
				became_ball = True

			if became_ball:
				for ball in self.balls:
					ball.body.setAngularVel((0.0,0.0,0.0))
					ball.body.setLinearVel((0.0,0.0,0.0))
		

	def generate_balls(self,amount):
		uid = glGenLists(amount)

		for i in range(amount):
			# generate random position
			x = random.uniform(-self.ball_area*3,self.ball_area*3)
			y = random.uniform(-self.ball_area*3,self.ball_area*3)
			z = random.uniform(-self.ball_area*3,self.ball_area*3)

			# generate random colors
			r = random.uniform(0,1)
			g = random.uniform(0,1)
			b = random.uniform(0,1)

			# generate display lists
			ball = Sphere((x,y,z),(r,g,b))
			ball.uid = uid + i
			ball.create_body(self.space,self.ode_world)
			#ball.body.setForce(scalp(minus_vector((0.0,0.0,0.0),ball.body.getPosition()),1000))

			glNewList(ball.uid,GL_COMPILE)
			ball.draw()
			glEndList()

			self.balls.append(ball)

	def calculate_shot(self,mpos):
		global curShot

		if not self.allStopped:
			return

		x = mpos[0]
		y = mpos[1]

		(cam_x,cam_y,cam_z) = self.cam_coords

		click_vector = (x/HALF_SIZE[0] - 1, 1 - y/HALF_SIZE[1], 0)
		#print click_vector
		inward_vector = (-cam_x,-cam_y,-cam_z)

		#inward_vector = (0,-cam_y,0)

		# FIXME: Shot vector combination is wrong. Do further geometric computations before coding
		#shot_vector = (click_vector[0] + inward_vector[0], click_vector[1] + inward_vector[1], inward_vector[2])
		shot_vector = inward_vector
		shot_norm = normalize_vector(shot_vector)

		# generate display list
		#(sx,sy,sz,ux,uy,uz) = from_spherical(self.rho - 2.0, self.theta, self.phi)
		(sx,sy,sz,ux,uy,uz) = from_rectangle(self.PCAM_Y - 2.0, self.recx, self.recz)

		# new additions
		up_shot = (ux,uy,uz)
		inward_norm = normalize_vector(inward_vector)
		cross_shot = crossp(up_shot,inward_norm)

		# calculate rotation
		rotated = rotatev(inward_norm,up_shot,(math.pi / 5) * -click_vector[0])
		rotated = rotatev(rotated,cross_shot,(math.pi / 5) * -click_vector[1])

		shot = Shot((sx,sy,sz),(1,0,0))
		shot.uid = glGenLists(1)
		curShot = shot.uid
		print "current: ", curShot
		#shot.direction = shot_norm
		shot.direction = rotated
		shot.create_body(self.space,self.ode_world)
		shot.body.setForce(scalp(shot.direction,self.shot_speed))
		shot.speed = self.shot_speed
		shot.geom.uid = shot.uid	# TODO
		shot.geom.type = 'shot'

		glNewList(shot.uid,GL_COMPILE)
		shot.draw()
		glEndList()

		self.shots.append(shot)


	def scroll_control(self,event):
		but = event.button
		if but == 4:	# up scroll
			self.shot_speed += SPEED_INC
			print self.shot_speed
		else:   # down scroll
			self.shot_speed -= SPEED_INC
			print self.shot_speed

	def click_control(self,event):
		if not self.wasdown:
			self.wasdown = True
			self.calculate_shot(pygame.mouse.get_pos())
	def key_control(self,event):
		pressed = pygame.key.get_pressed()

		if event.type == pygame.MOUSEBUTTONUP:
			self.wasdown = False

        	if pressed[K_a]:
	    		#self.theta += TURN_INC
	    		self.recx += RECT_TURN_INC
        	elif pressed[K_d]:
	    		#self.theta -= TURN_INC
	    		self.recx -= RECT_TURN_INC
        	if pressed[K_w]:
	    		#self.phi += TURN_INC
	    		self.recz += RECT_TURN_INC
        	elif pressed[K_s]:
	    		#self.phi -= TURN_INC
	    		self.recz -= RECT_TURN_INC
        	if pressed[K_g]:
	    		#self.rho += ZOOM_INC
			self.PCAM_Y += ZOOM_INC
        	elif pressed[K_t]:
	    		#self.rho -= ZOOM_INC
			self.PCAM_Y -= ZOOM_INC

		if self.recx >= WORLD_LX/2:
			self.recx = WORLD_LX/2 - 1
		elif self.recx <= -WORLD_LX/2:
			self.recx = -WORLD_LX/2 + 1

		if self.recz >= WORLD_LX/2:
			self.recz = WORLD_LX/2 - 1
		elif self.recz <= -WORLD_LX/2:
			self.recz = -WORLD_LX/2 + 1


	# find the distance to closest object (plane/ball e.g.) for a shot
	def closestDist(self,shot):
		point = shot.body.getPosition()
		ds = []

		# dist to planes
		for i in range(4):
			dist = point_to_plane(point,self.world.norm[i],-self.world.lx/2)
			ds.append(dist)

		# dist along Y axis
		for i in range(4,6):
			dist = point_to_plane(point,self.world.norm[i],-self.world.ly/2)
			ds.append(dist)

		# dis to shots
		for s in self.shots:
			if s.uid != shot.uid:
				dist = vector_mag(minus_vector(point,s.body.getPosition()))
				dist = dist - s.radius
				ds.append(dist)

		return min(ds)
		

	# NOT USED
	def updateShots(self):
		global curShot

		stopCheck = True

		for shot in self.shots:
			'''
			if shot.speed-FRICTION > 0:
				shot.speed -= FRICTION
				#shot.move(shot.direction)
				shot.body.addForce(scalp(shot.direction,shot.speed))
			else:
				shot.body.setForce((0,0,0))
			'''

			v = (vx,vy,vz) = shot.body.getLinearVel()
			mag = vector_mag(v)

			ag = vector_mag(shot.body.getAngularVel())
			mag = max(mag,ag)

			if shot.stopper > 0 and mag > STOP_THRES:
				shot.body.addForce(scalp(normalize_vector((-vx,-vy,-vz)),FRICTION))
				shot.stopper -= 1
				stopCheck = False
			elif not shot.stopper:
				shot.body.setLinearVel((0,0,0))
				shot.body.setAngularVel((0,0,0))
				shot.body.setTorque((0,0,0))
				shot.body.setForce((0,0,0))

				shot.stopper = 100

				# reset current shot
				if shot.uid == curShot:
					self.process = True
					curShot = -1

				self.stopShots.append(shot)

		self.allStopped = stopCheck


		if self.allStopped and self.process:
			#for shot in self.stopShots:	
			while self.stopShots:
				shot = self.stopShots.pop()
				shot.set_radius(max(self.closestDist(shot)-1,SHOT_RADIUS))
				#print shot.radius

				shot.geom = ode.GeomSphere(self.space,shot.radius)
				shot.geom.setBody(shot.body)

				shot.geom.uid = shot.uid
				shot.geom.type = 'shot'

				glNewList(shot.uid,GL_COMPILE)
				shot.draw()
				glEndList()

			self.process = False



	def killShots(self):
		for shot in self.shots:
			if remShots.has_key(shot.uid) and remShots[shot.uid] >= SHOT_LIFE:
				remShots.pop(shot.uid)	# delete
				self.shots.remove(shot)
				self.score += 1

	def goalCheck(self):
		for shot in self.shots:
			bpos = (bx,by,bz) = shot.body.getPosition()
			v = (vx,vy,vz) = shot.body.getLinearVel()
			f = (fx,fy,fz) = shot.body.getForce()
			#print bpos
			#print v, f
			#print by+shot.radius, " vs ", WORLD_LY/2 - GOAL_DIST, " vy: ", vy
			if (vy > 0 or (vy == 0 and vector_mag(f) == 0)) and by+shot.radius >= (WORLD_LY/2 - GOAL_DIST):
				print "GAME OVER! ",
				print "score: ", self.score
				return True
		return False
				


	def physics(self):
		self.space.collide((self.ode_world,self.contactgroup), near_callback)
		self.ode_world.quickStep(1.0)
		self.contactgroup.empty()

		'''
		for ball in self.balls:
			diff = minus_vector(self.com,ball.body.getPosition())
			#d = vector_mag(diff)	# gives distance
			ball.body.addForce(scalp(normalize_vector(diff),CG))
		'''

		for ball in self.balls:
			for ball2 in self.balls:
				if ball.uid == ball2.uid:
					continue
			diff = minus_vector(ball.body.getPosition(),ball2.body.getPosition())
			idiff = invert_vector(diff)
			d = vector_mag(diff)	# gives distance
			print vector_mag(diff)
			if not d:
				continue
			ball2.body.addForce(scalp(normalize_vector(diff),G/d**2))
			ball.body.addForce(scalp(normalize_vector(idiff),G/d**2))

		for ball in self.balls:
			diff = minus_vector((0.0,0.0,0.0),ball.body.getPosition())
			d = vector_mag(diff)	# gives distance
			if d > 10.0:
				bm = ball.body.getMass().mass
				ball.body.addForce(scalp(normalize_vector(diff),CG))

		'''
		for shot in self.shots:
			diff = minus_vector(self.com,shot.body.getPosition())
			d = vector_mag(diff)	# gives distance
			sm = shot.body.getMass().mass
			shot.body.addForce(scalp(normalize_vector(diff),G*self.total_mass*sm/d**2))
		'''

		#self.calculate_com()
		self.t_step += 1

	def netLines(self,yp,k):
		glBegin(GL_LINES)

		'''
		glVertex3f(self.world.lx/2,yp,0)
		glVertex3f(-self.world.lx/2,yp,0)
		glVertex3f(0,yp,self.world.lx/2)
		glVertex3f(0,yp,-self.world.lx/2)
		'''

		for i in range(k):
			glVertex3f(-self.world.lx/2+i*self.world.lx/k,yp,-self.world.lz/2)
			glVertex3f(-self.world.lx/2+i*self.world.lx/k,yp,self.world.lz/2)

		for i in range(k):
			glVertex3f(-self.world.lx/2,yp,-self.world.lz/2+i*self.world.lz/k)
			glVertex3f(self.world.lx/2,yp,-self.world.lz/2+i*self.world.lz/k)

		glEnd()

	def renderScene(self):
        	# Clear the screen, and z-buffer
        	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		#(cam_x,cam_y,cam_z,up_x,up_y,up_z) = from_spherical(self.rho,self.theta,self.phi)
		(cam_x,cam_y,cam_z,up_x,up_y,up_z) = from_rectangle(self.PCAM_Y,self.recx,self.recz)
		self.cam_coords = (cam_x,cam_y,cam_z)

		glLoadIdentity()
		#gluLookAt(cam_x,cam_y,cam_z,0.0,0.0,0.0,up_x,up_y,up_z)
		gluLookAt(cam_x,cam_y,cam_z,0.0,-cam_y,0.0,up_x,up_y,up_z)

		# Light must be transformed as well
		glLight(GL_LIGHT0, GL_POSITION,  (0.35, 0.8, 0.1, 0))	# old: (0,1,0,0)


    		glLightfv(GL_LIGHT1, GL_POSITION, (-0.35,-0.4,-0.1,0) )	# old: (0,-1,0,0)

    		#glMaterial(GL_FRONT, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))   
    		glMaterial(GL_FRONT, GL_AMBIENT, (0.2, 1.0, 0.2, 1.0))    
    		glMaterial(GL_FRONT, GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))

		# net
		#self.netLines(self.world.ly/2-GOAL_DIST,20)
		glCallList(self.net_uid)

		# draw goal lines
    		glMaterial(GL_FRONT, GL_AMBIENT, (0.2, 1.0, 0.2, 1.0))    
    		glMaterial(GL_FRONT, GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))
		glBegin(GL_LINE_LOOP)
		glVertex3f(-self.world.lx/2,self.world.ly/2-GOAL_DIST,-self.world.lz/2)
		glVertex3f(self.world.lx/2,self.world.ly/2-GOAL_DIST,-self.world.lz/2)
		glVertex3f(self.world.lx/2,self.world.ly/2-GOAL_DIST,self.world.lz/2)
		glVertex3f(-self.world.lx/2,self.world.ly/2-GOAL_DIST,self.world.lz/2)
		glEnd()

		#self.sphere.render()
		for ball in self.balls:
			ball.render()
		self.world.render()

		for shot in self.shots:
			shot.render()

tid = 0

def handler(frozenGame):
	global tid
	tid = os.getpid()

	while True:
		time.sleep(0.01)
		frozenGame.physics()

def run():
    
    glutInit()

    pygame.init()
    screen = pygame.display.set_mode(SCREEN_SIZE, HWSURFACE|OPENGL|DOUBLEBUF)
    
    frozenGame = Game()

    resize(*SCREEN_SIZE)
    init()
    
    clock = pygame.time.Clock()  

    t = threading.Thread(target=handler,args=(frozenGame,))
    t.start()

 
    while True:
	events = pygame.event.get()
	for event in events:
        	if event.type == QUIT:
    			import os
    			os.popen("kill -9 "+str(tid))
            		break
        	if event.type == KEYUP and event.key == K_ESCAPE:
    			import os
    			os.popen("kill -9 "+str(tid))
            		break
		if event.type == MOUSEBUTTONDOWN and event.button in [4,5]:
			frozenGame.scroll_control(event)
		if event.type == MOUSEBUTTONDOWN and event.button == 1:
			frozenGame.click_control(event)
	frozenGame.key_control(event)

        #time_passed = clock.tick()
        #time_passed_seconds = time_passed / 1000.

	frozenGame.killShots()
	frozenGame.updateShots()

	gameStatus = frozenGame.goalCheck()
	if gameStatus:
    		import os
    		os.popen("kill -9 "+str(tid))
            	break

	frozenGame.renderScene()
	#frozenGame.physics()

        # Show the screen
        pygame.display.flip()

	clock.tick(60)

    # end game actions

run()
