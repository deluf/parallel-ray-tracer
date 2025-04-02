class Sphere{
  Vector pos;
  float r;
  //reflection power
  float[] ks;
  //absorption power
  float[] kd;
  
  Sphere(Vector pos, float r, float[] kd, float[] ks){
    this.r = r;
    this.pos = pos.copy();
    this.kd = kd.clone();
    this.ks = ks.clone();
  }
}

Sphere[] loadSpheres(String file){
  Sphere spheres[];
  String[] s = loadStrings(file+".obj");
  spheres = new Sphere[s.length];
  for(int i = 0; i < s.length; i++){
    float[] data = float(split(s[i], ' '));
    Vector pos = new Vector(data[0], data[1], data[2]);
    float r = data[3];
    float[] kd = {data[4], data[5], data[6]};
    float[] ks = {data[7], data[8], data[9]};
    spheres[i] = new Sphere(pos, r, kd, ks);
  }
  return spheres;
}
