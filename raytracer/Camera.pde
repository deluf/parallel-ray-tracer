class Camera{
  Vector pos;
  Vector rot;
  float fov;
  
  Camera(float x, float y, float z, float FOV){
    pos = new Vector(x, y, z);
    rot = new Vector(0, 0, 0);
    fov = 1/tan(FOV/2);
  }
  
  void rotate(Vector p){
    rotateY(p);
    rotateX(p);
    rotateZ(p);
  }
  
  void rotateX(Vector p){
    Vector tmp = p.copy();
    p.y = tmp.y*cos(rot.x)-tmp.z*sin(rot.x);
    p.z = tmp.y*sin(rot.x)+tmp.z*cos(rot.x);
  }
  
  void rotateY(Vector p){
    Vector tmp = p.copy();
    p.x = tmp.x*cos(rot.y)+tmp.z*sin(rot.y);
    p.z = -tmp.x*sin(rot.y)+tmp.z*cos(rot.y);
  }
  
  void rotateZ(Vector p){
    Vector tmp = p.copy();
    p.x = tmp.x*cos(rot.z)-tmp.y*sin(rot.z);
    p.y = tmp.x*sin(rot.z)+tmp.y*cos(rot.z);
  }
  
  Vector[] calculateScreenCoords(){
    Vector[] out = new Vector[3];
    out[0] = new Vector(-1, fov, 1);
    out[1] = new Vector(1, fov, 1);
    out[2] = new Vector(-1, fov, -1);
    
    rotate(out[0]);
    rotate(out[1]);
    rotate(out[2]);
    
    //translate using camera coordinates;
    out[0].add(cam.pos);
    out[1].add(cam.pos);
    out[2].add(cam.pos);
    
    return out;
  }
}
