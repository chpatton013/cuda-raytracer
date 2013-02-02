<?php

Generator::main();

class Generator {
   public static function main() {
      $num_spheres = 100;
      $z_near = -10;
      $z_far = -100;
      $hfov = 60 * M_PI / 180;
      $min_radius = 0.1;
      $max_radius = 2;

      for ($ndx = 0; $ndx < $num_spheres; ++$ndx) {
         $z = self::unit_rand() * ($z_far - $z_near) + $z_near;
         $max_x = tan($hfov / 2) * $z;
         $x = self::unit_rand() * (2 * $max_x) - $max_x;
         $max_y = tan($hfov / 2) * $z;
         $y = self::unit_rand() * (2 * $max_y) - $max_y;

         $r = self::unit_rand() * ($max_radius - $min_radius) + $min_radius;

         $a = self::rand_vec();
         $d = self::rand_vec();
         $s = self::rand_vec();
         $shine = self::unit_rand() * 16;

         $sphere = array(
            'position' => array($x, $y, $z),
            'radius' => $r,
            'ambient' => $a,
            'diffuse' => $d,
            'specular' => $s,
            'shine' => $shine
         );

         self::print_sphere($sphere);
         echo "\n";
      }
   }

   private static function unit_rand() {
      return rand(0, 1000) / 1000;
   }
   private static function rand_vec() {
      return array(
         self::unit_rand(),
         self::unit_rand(),
         self::unit_rand()
      );
   }

   private static function print_vec($vec) {
      foreach ($vec as $val) {
         echo "$val ";
      }
   }
   private static function print_light($light) {
      self::print_vec($light['postiion']);
      echo "\n";
      self::print_vec($light['color']);
   }
   private static function print_sphere($sphere) {
      self::print_vec($sphere['position']);
      echo "\n";
      echo "{$sphere['radius']}\n";
      self::print_vec($sphere['ambient']);
      echo "\n";
      self::print_vec($sphere['diffuse']);
      echo "\n";
      self::print_vec($sphere['specular']);
      echo "\n";
      echo "{$sphere['shine']}\n";
   }
}
