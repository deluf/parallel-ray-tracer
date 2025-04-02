static class Vector{
  float x;
  float y;
  float z;
  
  Vector(float x, float y, float z){
    this.x = x;
    this.y = y;
    this.z = z;
  }
  
  
  Vector copy(){
    return new Vector(this.x, this.y, this.z);
  }
  
  static float dot(Vector v1, Vector v2){
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
  }
  
  static Vector mult(Vector v, float val){
    return new Vector(v.x*val, v.y*val, v.z*val);
  }
  
  static Vector mult(float val, Vector v){
    return new Vector(v.x*val, v.y*val, v.z*val);
  }
  
  static Vector add(Vector v1, Vector v2){
    return new Vector(v1.x+v2.x, v1.y+v2.y, v1.z+v2.z);
  }
  
  static Vector sub(Vector v1, Vector v2){
    return new Vector(v1.x-v2.x, v1.y-v2.y, v1.z-v2.z);
  }
  
  static float dist(Vector v1, Vector v2){
    return sqrt((v1.x-v2.x)*(v1.x-v2.x) + (v1.y-v2.y)*(v1.y-v2.y) + (v1.z-v2.z)*(v1.z-v2.z));
  }
  
  void add(Vector v){
    x += v.x;
    y += v.y;
    z += v.z;
  }
  
  Vector mult(float val){
    x *= val;
    y *= val;
    z *= val;
    return this;
  }
  
  Vector div(float val){
    x /= val;
    y /= val;
    z /= val;
    return this;
  }
  
  Vector cross(Vector v){
    return new Vector(this.y*v.z-this.z*v.y, this.z*v.x-this.x*v.z, this.x*v.y-this.y*v.x);
  }
  
  float mag(){
    return sqrt(x*x + y*y + z*z);
  }
  
  Vector normalize(){
    return this.div(this.mag());
  }
  
}
