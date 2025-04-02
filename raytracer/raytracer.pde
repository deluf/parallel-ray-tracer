int LIMIT = 1;

float FOV = PI/4;

String dir = "test4";

int THREAD_GRID = 4;

Camera cam = new Camera(0, -10, 0, FOV);


Sphere[] spheres; 
Triangle[] triangles;

AmbientLight ambLight = new AmbientLight(0.1, 0.1, 0.1);
DirLight[] dirLights;
PointLight[] pointLights;

Skybox skybox;

boolean showData = true;

int SSAA = 1;

void setup(){
  spheres = loadSpheres(dir+"/spheres");
  triangles = loadTriangles(dir+"/triangles");
  dirLights = loadDirLights(dir+"/dirLights");
  pointLights = loadPointLights(dir+"/pointLights");
  skybox = loadSkybox(dir+"/skybox");
  size(500, 500);
  stroke(255);
  loadPixels();
}

void draw(){
  Vector[] points = cam.calculateScreenCoords();
  Vector ul = points[0];
  Vector ur = points[1];
  Vector dl = points[2];
  
  Vector incX = Vector.sub(ur, ul).div(width);
  Vector incY = Vector.sub(dl, ul).div(height);
  
  Vector origin = cam.pos.copy();
  
  Thread[] threads = new Thread[THREAD_GRID*THREAD_GRID];
  
  for(int i = 0; i < THREAD_GRID; i++){
    for(int j = 0; j < THREAD_GRID; j++){
      threads[j + i*THREAD_GRID] = new Thread( new RayTracer(j*width/THREAD_GRID, (j+1)*width/THREAD_GRID, i*height/THREAD_GRID, (i+1)*height/THREAD_GRID, origin, ul, incX, incY) );
      threads[j+ i *THREAD_GRID].start();
    }
  }
  
  try {
    for(int i = 0; i < threads.length; i++)
      threads[i].join();
  } catch (InterruptedException e) {
      e.printStackTrace();
  }
  updatePixels();
  
  if(showData){
    String s = str(int(frameRate)) + "\nSSAA: x" + str(SSAA);
    text(s, 20, 20);
  }
}

void keyPressed(){
  //CAMERA POSITION
  Vector move = new Vector(0, 0, 0);
  if(key == 'a')
    move.x = -0.2;
  if(key == 'd')
    move.x = 0.2;
  if(key == 's')
    move.y = -0.2;
  if(key == 'w')
    move.y = 0.2;
  if(key == ' ')
    cam.pos.z += 0.2;
  if(keyCode == SHIFT)
    cam.pos.z -= 0.2;
    
  //CAMERA ROTATION
  if(key == 'q')
    cam.rot.y -= 0.1;
  if(key == 'e')
    cam.rot.y += 0.1;
  if(keyCode == UP)
    cam.rot.x += 0.1;
  if(keyCode == DOWN)
    cam.rot.x -= 0.1;
  if(keyCode == RIGHT)
    cam.rot.z -= 0.1;
  if(keyCode == LEFT)
    cam.rot.z += 0.1;
    
  cam.rotateZ(move);
  cam.pos.add(move);
}

void keyReleased(){
  if(key == '1'){
    saveFrame("save/raytracer-"+str(frameCount)+".png");
    println("IMAGE SAVED");
  }
  if(key == '2'){
    showData = !showData;
  }
  if(key == '-'){
    SSAA = constrain(SSAA-1, 1, 16);
  }
  if(key == '='){
    SSAA = constrain(SSAA+1, 1, 16);
  }
}
