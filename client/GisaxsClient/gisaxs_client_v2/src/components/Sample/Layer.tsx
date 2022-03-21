import {
  DeleteForever,
  ExpandLess,
  ExpandMore,
} from "@mui/icons-material";
import {
  Button,
  Card,
  CardActions,
  CardContent,
  Collapse,
  FormControl,
  Grid,
  TextField,
  Typography,
} from "@mui/material";
import { useState } from "react";

interface LayerProps {
  id: string;
  order: number;
  removeCallback: any;
}

const Layer = (props: LayerProps) => {
  const [collapsed, setCollapsed] = useState(true);

  const [thickness, setThickness] = useState("0");
  const [refBeta, setRefBeta] = useState("0");
  const [refDelta, setRefDelta] = useState("0");

  const handleButtonClick = () => {
    setCollapsed(!collapsed);
  };

  const handleRemove = (event) => {
    props.removeCallback();
  };

  return (
    <Card key={props.id} sx={{}}>
      <CardContent>
        <Grid
          container
          sx={{
            paddingBottom: collapsed ? 0 : 2,
          }}
        >
          <Grid item xs={6}>
            <Typography
              sx={{ fontSize: 14 }}
              color="text.secondary"
              gutterBottom
            >
              Layer{" "}
              {collapsed
                ? `[${props.order}, ${thickness}, ${refBeta}, ${refDelta}]`
                : ""}
            </Typography>
          </Grid>
          <Grid item xs={3}>
            <Button size="small" onClick={handleButtonClick}>
              {collapsed ? <ExpandMore /> : <ExpandLess />}
            </Button>
          </Grid>
          <Grid item xs={3}>
            <Button size="small" onClick={handleRemove}>
              <DeleteForever />
            </Button>
          </Grid>
        </Grid>

        <Collapse in={!collapsed}>
          <FormControl>
            <Grid container direction={"row"} rowSpacing={1}>
              <Grid item xs={6}>
                <TextField
                  label="order"
                  variant="outlined"
                  inputProps={{
                    readOnly: true,
                    disabled: true
                  }}
                  defaultValue={props.order}
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  label="thickness"
                  onChange={(e) => setThickness(e.target.value)}
                  variant="outlined"
                  defaultValue={0}
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  label="refBeta"
                  onChange={(e) => setRefBeta(e.target.value)}
                  variant="outlined"
                  defaultValue={0}
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  label="refDelta"
                  onChange={(e) => setRefDelta(e.target.value)}
                  variant="outlined"
                  defaultValue={0}
                />
              </Grid>
            </Grid>
          </FormControl>
        </Collapse>
      </CardContent>
      <CardActions></CardActions>
    </Card>
  );
};

export default Layer;
