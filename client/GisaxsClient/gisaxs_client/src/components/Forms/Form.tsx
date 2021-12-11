import JSONForm from "@rjsf/material-ui";
import type { JSONSchema7 } from "json-schema";
import type { UiSchema } from "@rjsf/core";
import CustomObjectFieldTemplate from "./CustomObjectFieldTemplate"
import CustomObjectFieldTemplateSmall from "./CustomObjectFieldTemplateSmall"
import CustomObjectFieldTemplateCollapse from "./CustomObjectFieldTemplateCollapse"
import CustomObjectFieldTemplateLarge from "./CustomObjectFieldTemplateLarge"

const uiSchema: UiSchema = {
    
    shapes: {
        items: {
            "ui:ObjectFieldTemplate": CustomObjectFieldTemplateCollapse,
            
            parameters: {
                "ui:ObjectFieldTemplate": CustomObjectFieldTemplate,
                radius: {
                    "ui:ObjectFieldTemplate": CustomObjectFieldTemplate,
                    mean:
                    {
                        "ui:widget": "range",
                    },
                    stddev:
                    {
                        "ui:widget": "range",
                    }
                },
                height: {
                    "ui:ObjectFieldTemplate": CustomObjectFieldTemplate,
                    mean:
                    {
                        "ui:widget": "range",
                    },
                    stddev:
                    {
                        "ui:widget": "range",
                    }
                },
            },
            refindex: {
                "ui:ObjectFieldTemplate": CustomObjectFieldTemplate,
            },
            location: {
                "ui:ObjectFieldTemplate": CustomObjectFieldTemplateSmall,
                posX:
                {
                    "ui:widget": "range",
                },
                posY:
                {
                    "ui:widget": "range",
                },
                posZ:
                {
                    "ui:widget": "range",
                }
            }
        },
    }
};

const schema2: JSONSchema7 = {
    type: "object",
    properties: {
        shapes: {
            $ref: "#/definitions/shapes",
        }
    },
    definitions: {
        shapes: {
            type: "array",
            title: "Shape configuration",
            items: {
                anyOf: [
                    {
                        title: "Sphere",
                        required: [],
                        properties: {
                            parameters: {
                                $ref: "#/definitions/sphereParams",
                            },
                            refindex: {
                                $ref: "#/definitions/refindex",
                            },
                            location: {
                                $ref: "#/definitions/location",
                            }
                        },
                    },
                    {
                        title: "Cylinder",
                        required: [],
                        properties: {
                            parameters: {
                                $ref: "#/definitions/cylinderParams",
                            },
                            refindex: {
                                $ref: "#/definitions/refindex",
                            },

                            location: {
                                $ref: "#/definitions/location",
                            }
                        },
                    },
                ],
            },
        },
        sphereParams: {
            title: "",
            type: "object",
            properties: {
                radius: {
                    $ref: "#/definitions/radius",
                },
            },
        },
        cylinderParams: {
            title: "",
            type: "object",
            properties: {
                radius: {
                    $ref: "#/definitions/radius",
                },
                height: {
                    $ref: "#/definitions/height",
                },
            },
        },
        refindex: {
            title: "Refraction index",
            type: "object",
            properties: {
                delta: {
                    type: "number",
                    title: "delta",
                    default: 1e-3,
                },
                beta: {
                    type: "number",
                    title: "beta",
                    default: 1e-5,
                },
            },
        },
        location: {
            title: "Location",
            type: "object",
            properties: {
                posX: {
                    type: "number",
                    title: "posX",
                    default: 0,
                },
                posY: {
                    type: "number",
                    title: "posY",
                    default: 0,
                },
                posZ: {
                    type: "number",
                    title: "posZ",
                    default: 0,
                },
            },
        },
        radius: {
            type: "object",
            title: "radius",
            $ref: "#/definitions/parameter",
        },
        height: {
            type: "object",
            title: "height",
            $ref: "#/definitions/parameter",
        },
        parameter: {
            type: "object",
            properties: {
                mean: {
                    type: "integer",
                    default: 0,
                    minimum: 0,
                    maximum: 100,
                },
                stddev: {
                    type: "integer",
                    default: 0,
                    minimum: 0,
                    maximum: 100,
                }
            }
        }
    },
};


function Form(props: { callback: Function, formData: any }) {
    const changeHandler = (value: any) => {
        props.callback(value.formData);
    };

    return (
            <JSONForm
                children={true}
                schema={schema2}
                formData={props.formData}
                uiSchema={uiSchema}
                onChange={(value) => { changeHandler(value) }}
                />
    );
}

export default Form;
