from matplotlib import cm
from matplotlib import colors
import numpy as np

LEVELS = 30
NESTED = False

def cmap_to_str(cmap, levels = 30, indent_level = 0, nested = NESTED):
	out = "["
	arr = cmap(np.linspace(0, 1, levels))
	for row in arr:
		out += "\n" + "\t"*(indent_level + 1)
		if nested:
			out += "["
		str_row = ""
		for val in row:
			str_row += f"{val}, "
		if nested:
			out += str_row[:-2]
			out += "],"
		else:
			out += str_row
	out += "\n" + "\t" * indent_level + "]"
	return out


all_maps = {map.name: map for ele in dir(cm) if isinstance((map:=getattr(cm, ele)), colors.Colormap)}
with open("src/colormaps.rs", "w") as file:
	# file.write("use phf::phf_map;\n\n")
	file.write("#![allow(non_camel_case_types)]\n")
	file.write("use strum::EnumIter;\n")
	file.write(f"pub const LEVEL_COUNT: usize = {LEVELS};\n\n")
	if NESTED:
		shape = "[[f32; 4]; LEVEL_COUNT]"
	else:
		shape = f"[f32; {int(LEVELS*4)}]"
	# file.write(f"pub const MAPS: phf::Map<&'static str, {shape}> = " + "phf_map! {")
	file.write("#[derive(Clone, PartialEq, EnumIter, Debug, Copy)]\n")
	file.write("pub enum KnownMaps{\n\t")
	tmp = ""
	for name, map in all_maps.items():
		tmp += f"{name},\n\t"
	tmp = tmp[:-1]
	tmp += "}"
	file.write(tmp)
	file.write("\n\n")
	file.write("impl KnownMaps{\n\t")
	
	file.write(f"pub fn get_map(&self) -> {shape}" + "{\n\t\t")
	file.write("match self{\n\t\t\t")
	tmp = ""
	for name, map in all_maps.items():
		tmp += f"Self::{name} => {cmap_to_str(map, indent_level = 3)},"
		tmp += "\n\t\t\t"
	tmp = tmp[:-1]
	file.write(tmp)
	file.write("}\n\t")
	file.write("}\n\n\t")
	
	file.write(f"pub fn get_name(&self) -> &'static str" + "{\n\t\t")
	file.write("match self{\n\t\t\t")
	tmp = ""
	for name, map in all_maps.items():
		tmp += f"Self::{name} => \"{name}\","
		tmp += "\n\t\t\t"
	tmp = tmp[:-1]
	file.write(tmp)
	file.write("}\n\t")
	file.write("}\n")
	
	
	file.write("}")


