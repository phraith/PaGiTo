import { Add } from "@mui/icons-material";
import {
  Button,
  Card,
  CardContent,
  Grid,
  List,
  ListItem,
  Menu,
  MenuItem,
  Typography,
} from "@mui/material";
import React, { useEffect } from "react";
import Cylinder from "./Cylinder";
import Sphere from "./Sphere";
import { v4 as uuidv4 } from "uuid";

interface GisaxsShapesProps {
  jsonCallback: any;
}

const GisaxsShapes = (props: GisaxsShapesProps) => {
  const [shapes, setShapes] = React.useState<any>([]);
  const [anchorEl, setAnchorEl] = React.useState(null);
  const [jsonData, setJsonData] = React.useState({});

  useEffect(() => {

    let formattedShapes = Object.keys(jsonData).map((key) => jsonData[key])
    props.jsonCallback(formattedShapes, "shapes")
  }, [jsonData]);

  const removeShape = (id: string) => {
    setShapes((shapes) => shapes.filter((shape) => shape.props.id !== id));
    setJsonData((jsonData) => {
      delete jsonData[id];
      return { ...jsonData };
    });
  };

  const createJsonForSphere = (sphereJson, shapeId) => {
    setJsonData((jsonData) => {
      jsonData[shapeId] = sphereJson;
      return { ...jsonData };
    });
  };

  const addShape = (e) => {
    setAnchorEl(e.currentTarget);
  };

  const addSphere = () => {
    const myid = uuidv4();
    setShapes([
      ...shapes,
      <Sphere
        key={myid}
        id={myid}
        removeCallback={() => removeShape(myid)}
        jsonCallback={createJsonForSphere}
      />,
    ]);

    setAnchorEl(null);
  };

  const addCylinder = () => {
    const myid = uuidv4();
    setShapes([
      ...shapes,
      <Cylinder
        key={myid}
        id={myid}
        removeCallback={() => removeShape(myid)}
        jsonCallback={createJsonForSphere}
      />,
    ]);

    setAnchorEl(null);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  return (
    <Card style={{ maxHeight: 700, overflow: "auto" }}>
      <CardContent>
        <Grid container>
          <Grid item xs={8}>
            <Typography>GisaxsShapesConfig</Typography>
          </Grid>
          <Grid item xs={4}>
            <Button size="small" onClick={addShape}>
              <Add />
            </Button>
            <Menu
              anchorEl={anchorEl}
              keepMounted
              open={Boolean(anchorEl)}
              onClose={handleClose}
            >
              <MenuItem onClick={addSphere}>Sphere</MenuItem>
              <MenuItem onClick={addCylinder}>Cylinder</MenuItem>
            </Menu>
          </Grid>
        </Grid>
        <List>
          {shapes.map((value) => {
            return <ListItem key={value.props.id}>{value}</ListItem>;
          })}
        </List>
      </CardContent>
    </Card>
  );
};

export default GisaxsShapes;
