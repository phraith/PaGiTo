import {
  DeleteForever,
  ExpandLess,
  ExpandMore,
  Remove,
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
import { useEffect, useState } from "react";

interface SubstrateProps {
  jsonCallback: any;
}

const Substrate = (props: SubstrateProps) => {
  const [collapsed, setCollapsed] = useState(true);
  const [refBeta, setRefBeta] = useState(2e-8);
  const [refDelta, setRefDelta] = useState(6e-6);

  const handleButtonClick = () => {
    setCollapsed(!collapsed);
  };

  useEffect(() => {
    props.jsonCallback(
      {
        refraction: {
          delta: refDelta,
          beta: refBeta,
        }
      },
      "substrate"
    );
  }, [refBeta, refDelta]);

  return (
    <Card sx={{}}>
      <CardContent>
        <Grid
          container
          sx={{
            paddingBottom: collapsed ? 0 : 2,
          }}
        >
          <Grid item xs={10}>
            <Typography
              sx={{ fontSize: 14 }}
              color="text.secondary"
              gutterBottom
            >
              Substrate {collapsed ? `[${refBeta.toExponential()}, ${refDelta.toExponential()}]` : ""}
            </Typography>
          </Grid>
          <Grid item xs={2}>
            <Button size="small" onClick={handleButtonClick}>
              {collapsed ? <ExpandMore /> : <ExpandLess />}
            </Button>
          </Grid>
        </Grid>

        <Collapse in={!collapsed}>
          <FormControl>
            <Grid container direction={"row"} rowSpacing={1}>
              <Grid item xs={6}>
                <TextField
                  type="number"
                  label="refBeta"
                  onChange={(e) => setRefBeta(Number(e.target.value))}
                  variant="outlined"
                  defaultValue={refBeta}
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  type="number"
                  label="refDelta"
                  onChange={(e) => setRefDelta(Number(e.target.value))}
                  variant="outlined"
                  defaultValue={refDelta}
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

export default Substrate;
