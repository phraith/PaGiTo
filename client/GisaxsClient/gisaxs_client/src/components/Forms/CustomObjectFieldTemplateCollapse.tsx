import React from "react";

import Grid from "@material-ui/core/Grid";
import { makeStyles } from "@material-ui/core/styles";
import Collapse from "@material-ui/core/Collapse";
import ToggleButton from "@material-ui/lab/ToggleButton";
import ToggleButtonGroup from "@material-ui/lab/ToggleButtonGroup";
const useStyles = makeStyles({
  root: {
    marginTop: 10,
  },
});

const CustomObjectFieldTemplateCollapse = ({
  DescriptionField,
  description,
  TitleField,
  title,
  properties,
  required,
  uiSchema,
  idSchema,
}: {
  DescriptionField: any;
  description: any;
  TitleField: any;
  title: any;
  properties: any;
  required: any;
  uiSchema: any;
  idSchema: any;
}) => {
  const classes = useStyles();
  const [showBox, setShowBox] = React.useState(false);
  return (
    <>
      {(uiSchema["ui:title"] || title) && (
        <TitleField
          id={`${idSchema.$id}-title`}
          title={title}
          required={required}
        />
      )}
      {description && (
        <DescriptionField
          id={`${idSchema.$id}-description`}
          description={description}
        />
      )}
      <ToggleButtonGroup
        value={showBox}
        exclusive
        onChange={(_, value: boolean) => setShowBox(value)}
      >
        <ToggleButton value={true}>Show</ToggleButton>
        <ToggleButton value={false}>Hide</ToggleButton>
      </ToggleButtonGroup>
      <Collapse in={showBox}>
        <Grid container direction="row" spacing={2} >
          {properties.map((element: any, index: any) => (
            <Grid
              item
              xs={12}
              key={index}
              
            >
              {element.content}
            </Grid>
          ))}
        </Grid>
      </Collapse>
    </>
  );
};

export default CustomObjectFieldTemplateCollapse;
