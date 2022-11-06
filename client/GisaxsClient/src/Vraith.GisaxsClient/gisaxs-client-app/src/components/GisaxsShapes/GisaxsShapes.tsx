import Add from "@mui/icons-material/Add";
import Button from "@mui/material/Button"
import Card from "@mui/material/Card"
import CardContent from "@mui/material/CardContent"
import MenuItem from "@mui/material/MenuItem"
import Menu from "@mui/material/Menu"
import Typography from "@mui/material/Typography"
import List from "@mui/material/List"
import ListItem from "@mui/material/ListItem"
import React, { useEffect } from "react";
import Cylinder from "./Cylinder";
import Sphere from "./Sphere";
import { v4 as uuidv4 } from "uuid";
import {
  CylinderConfig,
  SetLocalStorageEntity,
  SphereConfig,
} from "../Utility/DefaultConfigs";
import Box from "@mui/material/Box/Box";

interface GisaxsShapesProps {
  jsonCallback: any;
  isSimulation: boolean;
}

const GisaxsShapes = (props: GisaxsShapesProps) => {
  const [shapes, setShapes] = React.useState<any>([]);
  const [anchorEl, setAnchor] = React.useState(null);
  const [jsonData, setJsonData] = React.useState({});

  const localStorageEntityName: string = "shapesConfig";
  const configFieldName: string = "shapes";

  useEffect(() => {
    let formattedShapes = Object.keys(jsonData).map((key) => jsonData[key]);
    props.jsonCallback(formattedShapes, configFieldName);
    SetLocalStorageEntity(formattedShapes, [], localStorageEntityName);
    console.log(jsonData)
  }, [jsonData]);

  useEffect(() => {
    let data = localStorage.getItem(localStorageEntityName);
    if (data !== null) {
      let shapesConfig = JSON.parse(data);
      let cachedShapes: any = [];
      for (var shape of shapesConfig) {
        switch (shape.type) {
          case "sphere":
            cachedShapes = [...cachedShapes, createSphere(shape)];
            break;
          case "cylinder":
            cachedShapes = [...cachedShapes, createCylinder(shape)];
            break;
        }
      }
      setShapes(cachedShapes);
    }
  }, []);

  const removeShape = (id: string) => {
    setShapes((shapes) => shapes.filter((shape) => shape.props.id !== id));
    setJsonData((jsonData) => {
      delete jsonData[id];
      return { ...jsonData };
    });
  };

  const createJsonForShape = (sphereJson, shapeId) => {
    setJsonData((jsonData) => {
      jsonData[shapeId] = sphereJson;
      return { ...jsonData };
    });
  };

  const addShape = (e) => {
    setAnchor(e.currentTarget);
  };

  const addSphere = () => {
    setShapes([...shapes, createSphere(SphereConfig)]);
    setAnchor(null);
  };

  const createSphere = (sphereConfig) => {
    const myid = uuidv4();
    return (
      <Sphere
        key={myid}
        id={myid}
        isSimulation={props.isSimulation}
        removeCallback={() => removeShape(myid)}
        jsonCallback={createJsonForShape}
        initialConfig={sphereConfig}
      />
    );
  };

  const addCylinder = () => {
    setShapes([...shapes, createCylinder(CylinderConfig)]);
    setAnchor(null);
  };

  const createCylinder = (cylinderConfig) => {
    const myid = uuidv4();
    return (
      <Cylinder
        id={myid}
        isSimulation={props.isSimulation}
        removeCallback={() => removeShape(myid)}
        jsonCallback={createJsonForShape}
        initialConfig={cylinderConfig}
      />
    );
  };

  const handleClose = () => {
    setAnchor(null);
  };

  return (
    <Card style={{ maxHeight: 700, overflow: "auto" }}>
      <CardContent >
        <Box display="flex" sx={{ flexDirection: "column" }}>
          <Box display="flex" justifyContent={"space-between"} sx={{ paddingBottom: 1 }}>
            <Typography>GisaxsShapesConfig</Typography>

            <Button size="small" onClick={addShape}>
              <Add />
            </Button>

          </Box >
          <Menu
            anchorEl={anchorEl}
            keepMounted
            open={Boolean(anchorEl)}
            onClose={handleClose}
          >
            <MenuItem onClick={addSphere}>Sphere</MenuItem>
            <MenuItem onClick={addCylinder}>Cylinder</MenuItem>
          </Menu>

          <List>
            {shapes.map((value) => {
              return <ListItem key={value.props.id}>{value}</ListItem>;
            })}
          </List>
        </Box>
      </CardContent>
    </Card>
  );
};

export default GisaxsShapes;
