class AmbientLight{
  float[] kl;
  
  AmbientLight(float r, float g, float b){
    this.kl = new float[3];
    this.kl[0] = r;
    this.kl[1] = g;
    this.kl[2] = b;
  }
}

class DirLight {
  Vector dir;
  float[] kl;
  
  DirLight(Vector dir, float[] kl){
    this.dir = dir;
    this.kl = kl.clone();
  }
}

DirLight[] loadDirLights(String file){
  DirLight[] lights;
  String[] s = loadStrings(file+".obj");
  lights = new DirLight[s.length];
  for(int i = 0; i < s.length; i++){
    float[] data = float(split(s[i], ' '));
    float[] kl = {data[3], data[4], data[5]};
    lights[i] = new DirLight(new Vector(data[0], data[1], data[2]).normalize(), kl);
  }
  return lights;
}

class PointLight{
  Vector pos;
  float[] kl;
  
  PointLight(Vector pos, float[] kl){
    this.pos = pos;
    this.kl = kl.clone();
  }
}

PointLight[] loadPointLights(String file){
  PointLight[] lights;
  String[] s = loadStrings(file+".obj");
  lights = new PointLight[s.length];
  for(int i = 0; i < s.length; i++){
    float[] data = float(split(s[i], ' '));
    float[] kl = {data[3], data[4], data[5]};
    lights[i] = new PointLight(new Vector(data[0], data[1], data[2]), kl);
  }
  return lights;
}
